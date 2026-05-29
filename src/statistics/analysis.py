'''Variance, confidence intervals, and significance tests.'''

import json
from collections import defaultdict
from pathlib import Path
from itertools import combinations
import scipy.stats as stats
import pandas as pd
import numpy as np

def run_significance_tests(raw_responses_path: str = 'results/raw_responses.jsonl', output_path: str = 'results/significance.json'):
    records = []
    with open(raw_responses_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            records.append(json.loads(line))
            
    df = pd.json_normalize(records)
    
    if 'metrics' in df.columns:
        metrics_df = pd.json_normalize(df['metrics'])
        df = df.drop('metrics', axis=1).join(metrics_df)
    if 'judge' in df.columns:
        judge_df = pd.json_normalize(df['judge'])
        judge_norm = pd.DataFrame()
        for col in judge_df.columns:
            judge_norm[f'judge_{col}'] = pd.to_numeric(judge_df[col], errors='coerce')
        df = df.drop('judge', axis=1).join(judge_norm)
        
    tasks = df['task'].dropna().unique()
    metrics = ['bleu', 'rouge_l', 'bertscore', 'judge_overall']
    
    results = {}
    
    for task in tasks:
        results[task] = {}
        task_df = df[df['task'] == task]
        techniques = task_df['technique'].unique()
        
        for metric in metrics:
            if metric not in task_df.columns: continue
            results[task][metric] = []
            
            for t1, t2 in combinations(techniques, 2):
                data1 = task_df[task_df['technique'] == t1][metric].dropna().values
                data2 = task_df[task_df['technique'] == t2][metric].dropna().values
                data1 = pd.to_numeric(data1, errors='coerce')
                data2 = pd.to_numeric(data2, errors='coerce')
                
                data1 = data1[~np.isnan(data1)]
                data2 = data2[~np.isnan(data2)]
                
                if len(data1) < 2 or len(data2) < 2:
                    continue
                    
                t_stat, p_val = stats.ttest_ind(data1, data2, equal_var=False)
                mean_diff = np.mean(data1) - np.mean(data2)
                
                results[task][metric].append({
                    'technique_1': t1,
                    'technique_2': t2,
                    't_statistic': float(t_stat) if not np.isnan(t_stat) else 0.0,
                    'p_value': float(p_val) if not np.isnan(p_val) else 1.0,
                    'significant': bool(p_val < 0.05) if not np.isnan(p_val) else False,
                    'mean_diff': float(mean_diff),
                    'winner': t1 if mean_diff > 0 and p_val < 0.05 else (t2 if mean_diff < 0 and p_val < 0.05 else 'tie')
                })
                
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
        
    print(f'Saved pairwise significance tests to {output_path}')

if __name__ == '__main__':
    run_significance_tests()
