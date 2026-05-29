'''Chart export and report assembly utilities.'''

import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path

def _load_json(filepath: str):
    with open(filepath, 'r') as f:
        return json.load(f)

def generate_charts(summary_path: str = 'results/summary.json', output_dir: str = 'results/charts'):
    os.makedirs(output_dir, exist_ok=True)
    summary_data = _load_json(summary_path)
    
    # Restructure for plotting
    records = []
    for key, value in summary_data.items():
        task = value['task']
        tech = value['technique']
        for metric in ['bleu', 'rouge_l', 'bertscore', 'judge_overall']:
            if metric in value:
                records.append({
                    'task': task,
                    'technique': tech,
                    'metric': metric,
                    'mean': value[metric]['mean'],
                    'err': value[metric]['mean'] - value[metric]['ci_low']
                })
                
    df = pd.DataFrame(records)
    
    # Plot per task and metric
    tasks = df['task'].dropna().unique()
    metrics = df['metric'].dropna().unique()
    
    sns.set_theme(style='whitegrid')
    
    for task in tasks:
        task_df = df[df['task'] == task]
        for metric in metrics:
            metric_df = task_df[task_df['metric'] == metric]
            if metric_df.empty: continue
            
            plt.figure(figsize=(10, 6))
            ax = sns.barplot(
                x='technique', y='mean', data=metric_df, 
                palette='viridis', capsize=0.1
            )
            
            # error bars
            x_pos = range(len(metric_df['technique']))
            y = metric_df['mean'].values
            yerr = metric_df['err'].values
            plt.errorbar(x_pos, y, yerr=yerr, fmt='none', c='black', capsize=5)
            
            plt.title(f'Performance of Prompt Techniques on {task.capitalize()} ({metric.upper()})')
            plt.ylabel(metric)
            plt.xlabel('Prompting Technique')
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            clean_metric = metric.replace('_', '-')
            plt.savefig(f'{output_dir}/{task}_{clean_metric}.png', dpi=300)
            plt.close()

def generate_markdown_report(summary_path: str = 'results/summary.json', sig_path: str = 'results/significance.json', output_path: str = 'results/final_report.md'):
    summary = _load_json(summary_path)
    
    try:
        sig = _load_json(sig_path)
    except FileNotFoundError:
        sig = {}

    tasks = {}
    for k, v in summary.items():
        task = v['task']
        if task not in tasks: tasks[task] = []
        tasks[task].append(v)
        
    with open(output_path, 'w') as f:
        f.write('# LLM Prompting Techniques Evaluation Report\n\n')
        
        for task, data in tasks.items():
            f.write(f'## Task: {task.capitalize()}\n\n')
            
            f.write('### Performance Summary\n')
            f.write('| Technique | BLEU | ROUGE-L | BERTScore | Judge Overall |\n')
            f.write('|---|---|---|---|---|\n')
            
            # Find highest judge
            best_tech = None
            best_score = -1
            
            for item in data:
                tech = item['technique']
                bleu = item.get('bleu', {}).get('mean', 0.0)
                rouge = item.get('rouge_l', {}).get('mean', 0.0)
                bert = item.get('bertscore', {}).get('mean', 0.0)
                judge = item.get('judge_overall', {}).get('mean', 0.0)
                
                f.write(f'| {tech} | {bleu:.3f} | {rouge:.3f} | {bert:.3f} | {judge:.2f} |\n')
                
                if judge > best_score:
                    best_score = judge
                    best_tech = tech
                    
            f.write(f'\n**Recommendation for {task.capitalize()}**: Based on mean Judge Overall scores, {best_tech} achieved the highest raw performance ({best_score:.2f}).\n\n')
            
            f.write('### Statistical Significance (Judge Overall)\n')
            if task in sig and 'judge_overall' in sig[task]:
                f.write('| Technique 1 | Technique 2 | p-value | Significant | Winner |\n')
                f.write('|---|---|---|---|---|\n')
                for test in sig[task]['judge_overall']:
                    p = test['p_value']
                    is_sig = 'Yes' if test['significant'] else 'No'
                    winner = test['winner']
                    f.write(f"| {test['technique_1']} | {test['technique_2']} | {p:.4f} | {is_sig} | **{winner}** |\n")
            else:
                f.write('*No significance data available.*\n')
                
            f.write('\n---\n\n')
            
    print(f'Saved comprehensive markdown report to {output_path}')

if __name__ == '__main__':
    generate_charts()
    generate_markdown_report()
