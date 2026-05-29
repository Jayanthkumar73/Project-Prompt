import json
import pandas as pd
from pathlib import Path

def generate_statistics_csv():
    summary_path = Path('results/summary.json')
    sig_path = Path('results/significance.json')
    out_path = Path('results/statistics.csv')
    
    if not summary_path.exists():
        print("summary.json not found")
        return
        
    with open(summary_path, 'r') as f:
        summary_data = json.load(f)
        
    try:
        with open(sig_path, 'r') as f:
            sig_data = json.load(f)
    except FileNotFoundError:
        sig_data = {}
        
    rows = []
    
    for key, data in summary_data.items():
        task = data['task']
        tech = data['technique']
        
        # Try to find a p-value for this tech against zero_shot (or just pick one)
        p_val = "N/A"
        if task in sig_data and 'judge_overall' in sig_data[task]:
            for test in sig_data[task]['judge_overall']:
                if test['technique_1'] == tech or test['technique_2'] == tech:
                    # just record the p_value of the first test involving this technique
                    p_val = round(test['p_value'], 4)
                    break
                    
        for metric in ['bleu', 'rouge_l', 'bertscore', 'judge_overall']:
            if metric in data:
                m_data = data[metric]
                rows.append({
                    'Task': task,
                    'Technique': tech,
                    'Metric': metric,
                    'Mean': m_data.get('mean', 0),
                    'Std': m_data.get('std', 0),
                    'Variance': round(m_data.get('std', 0)**2, 6),
                    'CI Lower': m_data.get('ci_low', 0),
                    'CI Upper': m_data.get('ci_high', 0),
                    'p-value': p_val
                })
                
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"Created {out_path}")

if __name__ == '__main__':
    generate_statistics_csv()
