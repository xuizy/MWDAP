# MWDAP: Distributed Weighted Digraph Augmentation

Simulation code for the paper:

> **Distributed graph augmentation protocols for weighted strong connectivity in multi-agent systems**  
> Guilherme Ramos, Diogo Poças, Sérgio Pequito

## Overview

This repository implements and compares three methods for the Minimum Weight
Digraph Augmentation Problem (MWDAP):

1. **Algorithm 1** — Distributed augmentation protocol (proposed in the paper)
2. **Frederickson–Ja'Ja'** — Centralized 2-approximation
3. **Exact ILP** — Multi-commodity flow formulation (tractable for small instances)

Three random graph models are supported: Erdős–Rényi, Euclidean, and DAG.

## Requirements

```
networkx
numpy
scipy
```

Install with:
```bash
pip install networkx numpy scipy
```

## Usage

### As a module

```python
from simulation import (
    generate_er_digraph,
    distributed_algorithm,
    frederickson_jaja,
    exact_ilp,
    verify_augmentation,
)

# Generate a random digraph
G, W = generate_er_digraph(n=30, p=0.1, seed=42)

# Run the distributed algorithm
edges, cost, rounds = distributed_algorithm(G, W)
assert verify_augmentation(G, edges)

# Compare with Frederickson-Ja'Ja'
fj_edges, fj_cost = frederickson_jaja(G, W)

# Compare with exact ILP (small instances only)
ilp_edges, ilp_cost = exact_ilp(G, W, time_limit=120)
```

### Full experiments

Open `mwdap_simulation.ipynb` and run all cells. The notebook:

1. Runs all three methods across node sizes 5–100 on ER, Euclidean, and DAG models
2. Prints summary tables with approximation ratios
3. Exports results to JSON
4. Estimates empirical complexity (Table 1 in the paper)
5. Generates TikZ/pgfplots code for the paper figures

**Note:** The full experiment suite takes several hours. Reduce `ILP_MAX_N`,
`SAMPLES`, or `NODE_SIZES` for faster runs.

## File structure

- `simulation.py` — Core implementations (graph generation, algorithms, ILP, batch runner)
- `mwdap_simulation.ipynb` — Notebook reproducing all experiments and figures

## Node labeling

Node labels are `0, 1, ..., n-1` (zero-indexed).

## License

See the accompanying paper for terms of use.
