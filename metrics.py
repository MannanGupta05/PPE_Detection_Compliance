import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

ground_truth = pd.read_csv(r"C:\Users\Mannan Gupta\Desktop\ground_truth_ppe.csv")
predictions = pd.read_csv(r"C:\Users\Mannan Gupta\Desktop\predictions_1.csv")

ground_truth = ground_truth.sort_values("image_name").reset_index(drop=True)
predictions = predictions.sort_values("image_name").reset_index(drop=True)

eval_columns = [
    "person_present", "helmet_present", "helmet_compliant",
    "gloves_present", "gloves_compliant",
    "safety_vest_present", "safety_vest_compliant",
    "safety_suit_present", "safety_suit_compliant"
]

presence_cols = [
    "person_present", "helmet_present", "gloves_present",
    "safety_vest_present", "safety_suit_present"
]

compliance_cols = [
    "helmet_compliant", "gloves_compliant",
    "safety_vest_compliant", "safety_suit_compliant"
]

presence_metrics = {"accuracy": [], "precision": [], "recall": [], "f1": []}
compliance_metrics = {"accuracy": [], "precision": [], "recall": [], "f1": []}

for col in eval_columns:
    y_true = ground_truth[col]
    y_pred = predictions[col]
    
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    print(f"\nMetrics for '{col}':")
    print(f"  Accuracy : {acc:.2f}")
    print(f"  Precision: {prec:.2f}")
    print(f"  Recall   : {rec:.2f}")
    print(f"  F1-Score : {f1:.2f}")
    
    if col in presence_cols:
        presence_metrics["accuracy"].append(acc)
        presence_metrics["precision"].append(prec)
        presence_metrics["recall"].append(rec)
        presence_metrics["f1"].append(f1)
    elif col in compliance_cols:
        compliance_metrics["accuracy"].append(acc)
        compliance_metrics["precision"].append(prec)
        compliance_metrics["recall"].append(rec)
        compliance_metrics["f1"].append(f1)

def print_group_metrics(group_name, metrics):
    print(f"\n{group_name.upper()} METRICS")
    print(f"Average Accuracy : {sum(metrics['accuracy']) / len(metrics['accuracy']):.2f}")
    print(f"Average Precision: {sum(metrics['precision']) / len(metrics['precision']):.2f}")
    print(f"Average Recall   : {sum(metrics['recall']) / len(metrics['recall']):.2f}")
    print(f"Average F1-Score : {sum(metrics['f1']) / len(metrics['f1']):.2f}")

print_group_metrics("Presence (Object Detection)", presence_metrics)
print_group_metrics("Compliance (Pose Estimation)", compliance_metrics)
