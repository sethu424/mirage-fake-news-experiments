###################################
#
# Script created to run through different available LLM options on the MiRAGe test data
#
# Author: Sethu
# Updated by: Gemini
###################################

from datasets import load_dataset
from openai import OpenAI
import time
import json
import os
from tenacity import retry, stop_after_attempt, wait_fixed
from dotenv import load_dotenv
from loguru import logger
import openai
import vertexai
from vertexai.generative_models import GenerativeModel, GenerationConfig
import re

# Load environment variables from .env file
load_dotenv()

@retry(stop=stop_after_attempt(3), wait=wait_fixed(10))
def get_predictions(client: object, model_name: str, batch_captions: list[str]) -> list[str]:
    """
    Generates predictions for a batch of captions using the specified model.

    Args:
        client: The instantiated client for the AI service (OpenAI, Vertex AI, or Ollama).
        model_name: The name of the model to use for predictions.
        batch_captions: A list of string captions to classify.

    Returns:
        A list of predictions ('0' for real, '1' for fake).
    """
    predictions = []
    # Clean model name for Ollama to get the actual model identifier
    ollama_model_name = model_name.split("ollama-")[-1] if "ollama-" in model_name else model_name


    if model_name == "gpt-4o" or "ollama-" in model_name: # OpenAI or Ollama
        for caption in batch_captions:
            response = client.chat.completions.create(
                model=ollama_model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant. Your only job is to determine if a news caption is real or AI-generated. You must respond with '0' if the caption is real or '1' if it is fake. Do not provide any other text or explanation."},
                    {"role": "user", "content": caption}
                ],
                temperature=0,
            )
            prediction = response.choices[0].message.content.strip().lower()
            # Additional cleaning for local models that might still be verbose
            match = re.search(r'\d', prediction)
            if match:
                predictions.append(match.group(0))
            else:
                predictions.append("error") # Fallback

    elif "gemini" in model_name: # Vertex AI Gemini
        generation_config = GenerationConfig(
            temperature=0,
            max_output_tokens=4
        )

        for caption in batch_captions:
            prompt = f"Is the following news caption real or AI-generated? Respond with ONLY 0 if real OR 1 if fake. Do not include any other text or explanation.\n\nCaption: {caption}"
            
            response = client.generate_content(
                prompt,
                generation_config=generation_config
            )

            match = re.search(r'\d', response.text)
            if match:
                prediction = match.group(0)
            else:
                prediction = "error"
            
            predictions.append(prediction)

    return predictions

def process(model_name:str, test_split_name: str) -> None:
    """
    Method to generate the predictions output using the input model_name and on the input test_split_name specified.
    """
    logger.debug(f"Testing for model '{model_name}' initiated....")
    dataset = load_dataset("anson-huang/mirage-news", split=test_split_name)
    logger.debug(f"Dataset '{test_split_name}' loaded....")

    client = None
    if model_name == "gpt-4o":
        client = OpenAI(timeout=900.0)
        logger.debug("OpenAI client instantiated for GPT-4o.")
    elif "ollama-" in model_name:
        # Connect to local Ollama server
        client = OpenAI(
            base_url='http://localhost:11434/v1',
            api_key='ollama', # required, but can be any string
            timeout=900.0
        )
        logger.debug(f"OpenAI client instantiated for Ollama model: {model_name}.")
    elif "gemini" in model_name:
        try:
            if not vertexai.context.global_context.project:
                 vertexai.init(project=os.getenv("GCP_PROJECT_ID"), location=os.getenv("GCP_LOCATION"))
            client = GenerativeModel(model_name)
            logger.debug(f"Vertex AI client initialized for model '{model_name}'.")
        except Exception as e:
            logger.error(f"Failed to initialize Vertex AI. Please check your GCP_PROJECT_ID and GCP_LOCATION in the .env file and ensure you are authenticated. Error: {e}")
            return

    BATCH_SIZE = 20
    results = []
    output_dir = f"{OUTPUT_FOLDER_NAME}/{model_name}"
    os.makedirs(output_dir, exist_ok=True)

    for i in range(0, len(dataset), BATCH_SIZE):
        end_index = min(i + BATCH_SIZE, len(dataset))
        batch_range = range(i, end_index)
        batch = dataset.select(batch_range).to_dict()
        batch_list = [dict(zip(batch, t)) for t in zip(*batch.values())]
        texts = [item['text'] for item in batch_list]
        labels = [item['label'] for item in batch_list]

        try:
            batch_results = get_predictions(client, model_name, texts)
            logger.debug(f'Batch results: {batch_results}')

            for text, label, pred in zip(texts, labels, batch_results):
                results.append({
                    "text": text,
                    "label": label,
                    "prediction": pred.strip()
                })
            logger.debug(f"Processed batch {i} to {end_index}")
        except Exception as e:
            logger.error(f"Error processing batch {i} to {end_index}: {e}")
            for text, label in zip(texts, labels):
                 results.append({"text": text, "label": label, "prediction": "error"})
            time.sleep(10)

    output_filename = f"{output_dir}/{model_name}_results_{test_split_name}.json"
    with open(output_filename, "w") as f:
        json.dump(results, f, indent=4)
    logger.debug(f"Inference complete. Results saved to {output_filename}")


if __name__ == "__main__":
    OUTPUT_FOLDER_NAME = "./outputs"

    # Define the models and test splits to run
    models_to_test = [
        #"gpt-4o",
        #"gemini-2.0-flash-lite",
        #"ollama-gemma3:4b", # Example for Gemma 2B
        #"ollama-mistral:7b" ,
        "cogito:8b"
    ]
    test_splits = ['test1_nyt_mj', 'test2_bbc_dalle', 'test3_cnn_dalle', 'test4_bbc_sdxl', 'test5_cnn_sdxl']

    for model in models_to_test:
        logger.info(f"--- Starting processing for model: {model} ---")
        for split in test_splits:
            process(model_name=model, test_split_name=split)
        logger.info(f"--- Finished processing for model: {model} ---")
