import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def build_visualizations():
    # Load raw responses for boxplot/lineplot
    records = []
    with open('results/raw_responses.jsonl', 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    
    # Simple json normalize already flattens
    df_raw = pd.json_normalize(records)
    
    # Convert judge cols to float if they failed parsing
    for col in [c for c in df_raw.columns if c.startswith('judge.')]:
        df_raw[col] = pd.to_numeric(df_raw[col], errors='coerce')
        
    df_raw = df_raw.rename(columns={
        'metrics.rouge_l': 'rouge_l', 
        'metrics.bleu': 'bleu',
        'metrics.bertscore': 'bertscore',
        'judge.overall': 'judge_overall'
    })
    
    # 1. Bar Chart (rouge_bar.png)
    plt.figure(figsize=(10,6))
    sns.barplot(data=df_raw, x='technique', y='rouge_l', hue='task')
    plt.title('Average ROUGE-L Score Per Technique')
    plt.tight_layout()
    plt.savefig('charts/rouge_bar.png', dpi=300)
    plt.close()
    
    # 2. Box Plot (variance_boxplot.png)
    plt.figure(figsize=(12,6))
    sns.boxplot(data=df_raw, x='technique', y='judge_overall', hue='task')
    plt.title('Variance and Consistency of Judge Scores')
    plt.tight_layout()
    plt.savefig('charts/variance_boxplot.png', dpi=300)
    plt.close()
    
    # 3. Heatmap (heatmap.png)
    # Aggregate max technique per task based on mean judge overall
    pivot = df_raw.pivot_table(index='task', columns='technique', values='judge_overall', aggfunc='mean')
    plt.figure(figsize=(8,5))
    sns.heatmap(pivot, annot=True, cmap='YlGnBu')
    plt.title('Heatmap: Task vs Prompt Technique (Judge Overall)')
    plt.tight_layout()
    plt.savefig('charts/heatmap.png', dpi=300)
    plt.close()
    
    # 4. Line Plot (stability_lineplot.png)
    plt.figure(figsize=(10,6))
    # FutureWarning: ci is deprecated, use errorbar=None
    sns.lineplot(data=df_raw, x='run_index', y='judge_overall', hue='technique', marker='o', errorbar=None)
    plt.title('Score Stability Across Runs')
    plt.tight_layout()
    plt.savefig('charts/stability_lineplot.png', dpi=300)
    plt.close()
    
    print("Exported 4 charts to charts/")

if __name__ == '__main__':
    build_visualizations()
