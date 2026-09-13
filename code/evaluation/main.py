"""
Evaluation Runner for Buy or Wait? Financial Decision Agent.

Runs the pipeline against sample_requests.csv to benchmark performance and metrics.
"""

from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parent.parent.parent
CODE_DIR = Path(__file__).resolve().parent.parent

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

try:
    from code.main import run_pipeline
except ImportError:
    from main import run_pipeline


def main():
    import pandas as pd
    dataset_dir = BASE_DIR / "dataset"
    requests_filename = "sample_requests.csv"
    output_path = CODE_DIR / "evaluation" / "sample_output.csv"
    report_path = CODE_DIR / "evaluation" / "sample_usage_report.md"

    print("[*] Running evaluation workflow on sample_requests.csv...")
    df = run_pipeline(
        dataset_dir=dataset_dir,
        requests_filename=requests_filename,
        output_csv_path=output_path,
        report_path=report_path,
    )
    print(f"[+] Pipeline executed: {len(df)} sample rows generated.")

    # Ground truth validation benchmark
    gt_path = dataset_dir / "sample_requests.csv"
    if gt_path.exists():
        gt_df = pd.read_csv(gt_path)
        merged = pd.merge(gt_df, df, on="request_id", suffixes=("_gt", "_pred"))
        
        status_matches = (merged["affordability_status_gt"] == merged["affordability_status_pred"]).sum()
        status_acc = (status_matches / len(merged)) * 100
        
        method_matches = (merged["recommended_payment_method_gt"] == merged["recommended_payment_method_pred"]).sum()
        method_acc = (method_matches / len(merged)) * 100
        
        print("\n========================================================")
        print("               EVALUATION WORKFLOW BENCHMARK             ")
        print("========================================================")
        print(f"Total Evaluated:              {len(merged)} sample requests")
        print(f"Affordability Status Accuracy: {status_acc:.1f}% ({status_matches}/{len(merged)})")
        print(f"Payment Method Accuracy:       {method_acc:.1f}% ({method_matches}/{len(merged)})")
        print("========================================================\n")


if __name__ == "__main__":
    main()
