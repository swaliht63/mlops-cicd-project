import json
from pathlib import Path

import joblib


ROOT_DIR = Path(__file__).resolve().parents[1]
BASELINE_PATH = ROOT_DIR / "config" / "production_baseline.json"
MODEL_PATH = ROOT_DIR / "artifacts" / "model.joblib"


def main():
    baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    candidate = joblib.load(MODEL_PATH)

    baseline_metrics = baseline["metrics"]
    candidate_metrics = candidate["metrics"]
    allowed_drop = float(baseline["maximum_allowed_metric_drop"])

    failures = []

    for metric_name, baseline_value in baseline_metrics.items():
        if metric_name not in candidate_metrics:
            failures.append(f"{metric_name}: missing from candidate")
            continue

        candidate_value = float(candidate_metrics[metric_name])
        minimum_allowed = float(baseline_value) - allowed_drop

        print(
            f"{metric_name}: baseline={baseline_value:.3f}, "
            f"candidate={candidate_value:.3f}, "
            f"minimum={minimum_allowed:.3f}"
        )

        if candidate_value < minimum_allowed:
            failures.append(
                f"{metric_name} regressed: "
                f"{candidate_value:.3f} < {minimum_allowed:.3f}"
            )

    if failures:
        raise SystemExit("Candidate rejected:\n- " + "\n- ".join(failures))

    print("Candidate passed baseline comparison.")


if __name__ == "__main__":
    main()
