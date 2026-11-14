# Project — Ensemble methods and PAC-Bayesian Learning

The subject is in `projet.pdf`.

- **Part 1** (`code/ensembles.ipynb`): our own implementations of bagging (with out-of-bag error), discrete AdaBoost and stacking, compared with random forests, gradient boosting and simple baselines on 18 real datasets.
- **Part 2** (`code/pac_bayes.ipynb`): PAC-Bayesian bounds (Catoni, McAllester, Seeger) and their minimization, using `code/pacbayes_tools.py`.

## Contents

| Path | What it is |
|------|-----------|
| `projet.pdf` | the subject (Part 1: ensemble methods; Part 2: PAC-Bayesian bounds) |
| `code/benchkit.py` | the datasets of the course: `real_dataset(name)` returns `X`, `y` with labels in {-1,+1}; `REAL_SMALL` and `REAL_LARGE` list the available names; plus simple simulators and an evaluation helper `evaluate` |
| `code/ensembles_lib.py` | Part 1: `MyBagging`, `MyAdaBoost`, `MyStacking`, the model catalog `make_models()` and the evaluation protocol `evaluate_model` |
| `code/ensembles.ipynb` | Part 1: experiments, benchmark and analysis |
| `code/pacbayes_tools.py` | Part 2, **complete code**: the three bounds (linear, McAllester, Seeger), the algorithms that minimize bounds A and B for stumps (categorical distributions) and linear classifiers (Gaussian distributions), and the functions `run_categorical` / `run_gaussian` that run everything for one dataset and one split |
| `code/pac_bayes.ipynb` | Part 2: experiments and analysis |
| `common/tptools.py`, `data/` | dataset loader and datasets (do not modify) |
| `results/` | outputs of the notebooks (CSV tables and figures) |
| `requirements.txt` | Python dependencies |

## Getting started

Requirements: Python 3.12 or later (the versions pinned in `requirements.txt` need it), `numpy`, `scipy`, `scikit-learn`, `matplotlib`, `pandas`, and `jupyter` (or VS Code with the Jupyter extension) to run the notebooks.

```bash
pip install -r requirements.txt
pip install jupyter        # only if you run the notebooks outside VS Code
```

Quick check of the provided code:

```bash
cd code
python benchkit.py         # quick check of the datasets and of the evaluation helper
python pacbayes_tools.py   # Part 2 example on wdbc: compare with the "Expected values" box of the subject
```

```python
import benchkit as bk
X, y = bk.real_dataset("wdbc")
print(X.shape, bk.REAL_SMALL + bk.REAL_LARGE)

from pacbayes_tools import run_categorical, run_gaussian, show
show("wdbc, stumps", run_categorical(X, y, seed=0))
```

The features are **not** standardized: standardize after splitting, using the training part only (this is what `evaluate_model` does).

## Reproducing the results

Both notebooks must be run from the `code/` folder (open them from there). Use *Restart kernel and Run All*, or from a terminal:

```bash
cd code
jupyter nbconvert --to notebook --execute ensembles.ipynb --inplace
jupyter nbconvert --to notebook --execute pac_bayes.ipynb --inplace
```

### Part 1 — `code/ensembles.ipynb`

All randomness is seeded (`seed=0`): splits, bootstrap samples, trees and label noise.

The benchmark (section 7) takes about 20 minutes with `N_REP = 10` and is cached in `results/benchmark.csv`: **delete this file to recompute it**, otherwise it is reloaded as is. Set `N_REP = 3` for a quick test (and delete the cached file afterwards). The other sections recompute and overwrite their output files at each run.

| Report item | Notebook section | Output |
|-------------|------------------|--------|
| Table 1: datasets (m, d, minority share) | 1 | displayed in the notebook |
| Table 2: comparison with scikit-learn (3 datasets) | 6 | `results/validation_sklearn.csv` |
| OOB error of our bagging vs scikit-learn | 6 | `results/oob_comparison.csv` |
| Benchmark of all methods (10 models x 18 datasets) | 7 | `results/benchmark.csv` |
| Table 3: average ranks per metric and group | 8 | `results/ranks_by_metric.csv` |
| Gap accuracy minus balanced accuracy | 8 | `results/acc_minus_bacc.csv`, `results/acc_minus_bacc.png` |
| Figure 1: average ranks and Nemenyi critical distance | 9 | `results/ranks_bacc.png` |
| Table 4: gain over a single tree / logistic regression | 10 | `results/gains_vs_tree.csv`, `results/gains_vs_logreg.csv` |
| Table 5: bagging vs boosting per dataset | 11 | `results/bagging_vs_boosting.csv` |
| Figure 2: robustness to label noise | 11 | `results/noise.png`, `results/noise.csv` |
| Table 6: depth of the base trees | 11 | `results/depth.csv` |
| Table 7: stacking, gain and cost | 12 | `results/stacking.csv`, `results/fit_times.csv` |
| Conclusions | 13 | in the notebook |

### Part 2 — `code/pac_bayes.ipynb`

*To be completed by the partner: table "report item / notebook section / output file" for questions 1 to 5.*

## Experimental protocol (Part 1)

- **Datasets**: 18 of the 22 datasets of the course (`bankmarketing`, `bupa`, `balance`, `splice` are left out). A dataset is called *imbalanced* when its minority class is below 30%. Multiclass datasets are binarized one-vs-rest in `data/prepdata.py`.
- **Splits**: 10 repeated stratified hold-out splits (70% train / 30% test), same seed for all models.
- **Preprocessing**: standardization with the training statistics only.
- **Metrics**: accuracy, balanced accuracy, F1 of the minority class, AUC.
- **Models**: single tree, logistic regression, 7-NN (baselines); our bagging, AdaBoost (depth 1 and 3) and stacking; random forest, `GradientBoostingClassifier`, `HistGradientBoostingClassifier`. Hyperparameters are scikit-learn defaults (no tuning).
- **Label noise**: applied to the training labels only; test labels stay clean.

## Authors

*[Madjiguène Fall]* (Part 1), *[Vo Nam Dang]* (Part 2); each part was reviewed by the other.