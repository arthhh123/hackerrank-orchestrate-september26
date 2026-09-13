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

from code.main import run_pipeline


def main():
    dataset_dir = BASE_DIR / "dataset"
    requests_filename = "sample_requests.csv"
    output_path = CODE_DIR / "evaluation" / "sample_output.csv"
    report_path = CODE_DIR / "evaluation" / "usage_report.md"

    print("[*] Running evaluation on sample_requests.csv...")
    df = run_pipeline(
        dataset_dir=dataset_dir,
        requests_filename=requests_filename,
        output_csv_path=output_path,
        report_path=report_path,
    )
    print(f"[+] Evaluation finished: {len(df)} sample rows evaluated.")


if __name__ == "__main__":
    main()
