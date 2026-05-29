from src.execution.runner import run_experiment
from src.prompting.prompt_catalog import SUPPORTED_TECHNIQUES

tasks = [
    ('summarization', 'data/summarization/test_cases.json'),
    ('code_generation', 'data/code_generation/test_cases.json'),
    ('reasoning', 'data/reasoning/test_cases.json')
]

for task, path in tasks:
    for tech in SUPPORTED_TECHNIQUES:
        if task == 'summarization' and tech == 'zero_shot':
            print("Skipping summarization -> zero_shot (Already compiled)")
            continue
        print(f"\n==================================================")
        print(f"Starting testing for {task.upper()} - {tech.upper()}")
        print(f"==================================================")
        run_experiment(path, task, tech, repeats=3)
