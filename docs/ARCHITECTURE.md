# Frozen Architecture

## Layers
1. `configs/`: experiment and model configuration.
2. `templates/` + `src/prompting/`: prompt registry and renderer.
3. `src/execution/`: model calls and experiment runner.
4. `src/evaluation/`: metric and judge evaluators.
5. `src/statistics/`: CI, variance, significance testing.
6. `src/reporting/`: notebooks/charts/report assembly.
7. `src/api/`: service endpoints.
8. `src/contracts/`: canonical schemas used by all layers.

## Core Principle
Generation, evaluation, and statistics are isolated modules connected through versioned artifacts.
