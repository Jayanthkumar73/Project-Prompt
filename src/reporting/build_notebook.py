import nbformat as nbf
from nbconvert import HTMLExporter

def create_and_export_notebook():
    nb = nbf.v4.new_notebook()
    
    nb.cells.append(nbf.v4.new_markdown_cell('# LLM Prompt Engineering Evaluation Report'))
    nb.cells.append(nbf.v4.new_markdown_cell('## 1. Introduction\nThis report analyzes the performance of 6 prompting techniques across Summarization, Code Generation, and Reasoning tasks.'))
    nb.cells.append(nbf.v4.new_markdown_cell('## 2. Methodology\nTechniques tested:\n- Zero-shot\n- Few-shot\n- Chain-of-Thought\n- Role-based\n- Tree-of-Thought\n- Structured Output'))
    nb.cells.append(nbf.v4.new_markdown_cell('## 3. Results Overview\nBelow are visualizations summarizing our empirical findings based on multiple independent runs.'))
    
    # Let's add code cell that would technically just show images or load pandas
    code = '''import pandas as pd
df = pd.read_csv('../results/statistics.csv')
print(df.groupby(['Task', 'Technique'])['Mean'].mean().unstack())
    '''
    nb.cells.append(nbf.v4.new_code_cell(code))
    
    nb.cells.append(nbf.v4.new_markdown_cell('## Visualizations\n### ROUGE-L Average Score per Technique'))
    nb.cells.append(nbf.v4.new_markdown_cell('<img src="../charts/rouge_bar.png" width="600" />'))
    
    nb.cells.append(nbf.v4.new_markdown_cell('### Consistency (Boxplot)'))
    nb.cells.append(nbf.v4.new_markdown_cell('<img src="../charts/variance_boxplot.png" width="600" />'))
    
    nb.cells.append(nbf.v4.new_markdown_cell('### Best Techniques Matrix (Heatmap)'))
    nb.cells.append(nbf.v4.new_markdown_cell('<img src="../charts/heatmap.png" width="600" />'))
    
    nb.cells.append(nbf.v4.new_markdown_cell('### Stability Across Runs (Lineplot)'))
    nb.cells.append(nbf.v4.new_markdown_cell('<img src="../charts/stability_lineplot.png" width="600" />'))
    
    nb.cells.append(nbf.v4.new_markdown_cell('## 4. Conclusions\nThe empirical data strongly highlights the consistency of prompt patterns compared across tasks.'))
    
    with open('notebooks/evaluation_report.ipynb', 'w') as f:
        nbf.write(nb, f)
        
    print('Created notebooks/evaluation_report.ipynb')
    
    # Export to HTML
    exporter = HTMLExporter()
    body, resources = exporter.from_notebook_node(nb)
    
    with open('report/evaluation_report.html', 'w') as f:
        f.write(body)
        
    print('Exported to report/evaluation_report.html')

if __name__ == '__main__':
    create_and_export_notebook()
