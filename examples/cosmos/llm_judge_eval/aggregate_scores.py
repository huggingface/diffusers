import json
import glob
import os
import sys
from collections import Counter

assert len(sys.argv)==2, "input the data dir"

def aggregate_pred_scores(output_dir):
    json_files = sorted(
        glob.glob(os.path.join(output_dir, "*.json")),
        key=lambda p: int(os.path.splitext(os.path.basename(p))[0])
    )

    all_scores = []
    per_file = []

    for path in json_files:
        with open(path) as f:
            data = json.load(f)
        scores = [v["pred_score"] for v in data.get("videos", []) if "pred_score" in v]
        all_scores.extend(scores)
        per_file.append((os.path.basename(path), scores, data.get("mean_pred_score")))

    print(f"{'File':<12} {'Scores':<30} {'Mean'}")
    print("-" * 60)
    for fname, scores, mean in per_file:
        formatted = [f"{s:.2f}" for s in scores]
        print(f"{fname:<12} {str(formatted):<30} {mean:.2f}")

    print("-" * 60)

    dist = Counter(all_scores)
    for score in sorted(dist):
        print(f"  {score}: {dist[score]}")

    overall_avg = sum(all_scores) / len(all_scores) if all_scores else 0
    print(f"\nTotal videos : {len(all_scores)}")
    print(f"Average score: {overall_avg:.2f}")
    return overall_avg


if __name__ == "__main__":
    output_dir = sys.argv[1]
    aggregate_pred_scores(output_dir)
