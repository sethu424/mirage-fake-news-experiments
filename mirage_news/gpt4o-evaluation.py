import json
import numpy as np
from miragenews.utils.metrics import calculate_metrics

# 1. Load results
with open("gpt4o_mini_flex_test_results.json", "r") as f:
    results = json.load(f)

# 2. Map predictions and labels to binary
def label_to_int(label):
    # Adjust if your ground truth uses different values
    if isinstance(label, str):
        return 1 if label.lower() == "fake" else 0
    return int(label)

def pred_to_int(pred):
    # Accepts various forms of model output
    pred = pred.strip().lower()
    if "fake" in pred:
        return 1
    elif "real" in pred:
        return 0
    else:
        # fallback: treat as 0 (real)
        return 0

y_true = np.array([label_to_int(item["label"]) for item in results])
y_pred = np.array([pred_to_int(item["prediction"]) for item in results])

# 3. Calculate metrics
# Since GPT-4o-mini gives hard predictions, use y_pred as probabilities for thresholding
metrics = calculate_metrics(y_true, y_pred, threshold=0.5)

# 4. Print and save results
print("Evaluation Metrics for GPT-4o-mini (flex):")
for k, v in metrics.items():
    print(f"{k}: {v}")

with open("gpt4o_mini_flex_eval_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)