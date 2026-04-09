# Black-Box Adversarial Attacks via Genetic Algorithms

> A black-box adversarial attack framework using Genetic Algorithms against CNN classifiers, with adversarial training defense.

## Overview

This project explores the use of **Genetic Algorithms (GA)** to generate adversarial examples against neural networks in a **black-box setting** — without access to model weights or gradients. We compare our approach against classical white-box attacks (FGSM, PGD) and evaluate adversarial training as a defense.

## Key Contributions

- Black-box adversarial attack using GA (no gradient access required)
- Comparison with FGSM and PGD baselines
- Adversarial training defense pipeline
- Analysis on CIFAR-10 dataset

## Project Structure

```
├── notebooks/
│   ├── 01_train_model.ipynb       # Train CNN on CIFAR-10
│   ├── 02_baseline_attacks.ipynb  # FGSM and PGD baselines
│   ├── 03_ga_attack.ipynb         # GA black-box attack
│   ├── 04_defense.ipynb           # Adversarial training
│   └── 05_results.ipynb           # Final tables and figures
├── src/
│   ├── model.py                   # CNN architecture
│   ├── ga.py                      # Genetic Algorithm
│   └── attacks.py                 # FGSM, PGD implementations
├── results/
│   ├── figures/                   # Generated plots
│   └── tables/                    # Result tables
└── paper/                         # LaTeX paper
```

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Run

Open notebooks in order: `01 → 02 → 03 → 04 → 05`

## Results

| Method    | Type       | Attack Success Rate | Avg L2 Perturbation |
|-----------|------------|--------------------|--------------------|
| FGSM      | White-box  | TBD                | TBD                |
| PGD       | White-box  | TBD                | TBD                |
| GA (ours) | Black-box  | TBD                | TBD                |

*Results will be updated after experiments.*

## Author

Aleksej Ersov — RN77/2024
