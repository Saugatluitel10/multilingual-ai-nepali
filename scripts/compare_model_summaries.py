"""
Aggregate evaluation summaries from all model checkpoint directories and generate a comparison table/plot.
"""
import os
import json
import pandas as pd
from pathlib import Path

SUMMARY_FILES = list(Path('models/checkpoints').rglob('eval_summary.json'))

rows = []
for f in SUMMARY_FILES:
    with open(f, 'r') as fin:
        summary = json.load(fin)
        row = {
            'model_name': summary.get('model_name'),
            'adapter': summary.get('adapter'),
            'best_metric': summary.get('best_metric'),
            'best_checkpoint': summary.get('best_model_checkpoint'),
            'timestamp': summary.get('timestamp'),
            'summary_path': str(f)
        }
        # Add last eval metrics if available
        if summary.get('metrics'):
            last_eval = next((m for m in reversed(summary['metrics']) if 'eval_loss' in m), None)
            if last_eval:
                for k, v in last_eval.items():
                    if isinstance(v, (float, int)):
                        row[k] = v
        rows.append(row)

df = pd.DataFrame(rows)
if not df.empty:
    print("\nModel Evaluation Comparison Table:\n")
    print(df[['model_name', 'adapter', 'best_metric', 'eval_loss', 'eval_accuracy', 'eval_f1', 'timestamp', 'summary_path']].to_markdown(index=False))
    # Optionally, save as CSV for dashboarding
    df.to_csv('models/checkpoints/model_comparison.csv', index=False)
else:
    print("No model summaries found in models/checkpoints/.")
