"""Part 1: our own ensemble methods and the evaluation protocol.

    MyBagging, MyAdaBoost, MyStacking    from-scratch implementations (labels in {-1,+1})
    default_base_models()                base models of the stacking
    make_models()                        all the models compared in the benchmark
    evaluate_model(model, X, y, ...)     repeated stratified hold-out (metrics + fit time)
"""
import time
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                              HistGradientBoostingClassifier)
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             f1_score, roc_auc_score)
from benchkit import flip_labels

__all__ = ["MyBagging", "MyAdaBoost", "MyStacking", "default_base_models",
           "make_models", "get_scores", "compute_metrics", "evaluate_model",
           "METRICS"]

METRICS = ["acc", "bacc", "f1_min", "auc"]


# ------------------------------------------------------------------ bagging
class MyBagging(BaseEstimator, ClassifierMixin):
    """Bagging of decision trees with out-of-bag error (attribute oob_error_)."""

    def __init__(self, n_estimators=100, max_depth=None, random_state=0):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state

    def fit(self, X, y):
        rng = np.random.default_rng(self.random_state)
        n = len(y)
        self.classes_ = np.array([-1, 1])
        self.trees_ = []
        votes = np.zeros(n)   # sum of OOB votes per point
        counts = np.zeros(n)  # number of OOB votes per point
        for _ in range(self.n_estimators):
            idx = rng.integers(0, n, n)              # bootstrap sample
            oob = np.setdiff1d(np.arange(n), idx)    # points not drawn
            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                random_state=int(rng.integers(1 << 31))).fit(X[idx], y[idx])
            self.trees_.append(tree)
            if len(oob):
                votes[oob] += tree.predict(X[oob])
                counts[oob] += 1
        seen = counts > 0
        self.oob_error_ = np.mean(np.sign(votes[seen]) != y[seen])
        return self

    def decision_function(self, X):
        return np.mean([t.predict(X) for t in self.trees_], axis=0)

    def predict(self, X):
        return np.where(self.decision_function(X) >= 0, 1, -1)


# ----------------------------------------------------------------- adaboost
class MyAdaBoost(BaseEstimator, ClassifierMixin):
    """Discrete AdaBoost, trees of depth max_depth as weak learners."""

    def __init__(self, n_estimators=100, max_depth=1, random_state=0):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state

    def fit(self, X, y):
        n = len(y)
        self.classes_ = np.array([-1, 1])
        w = np.full(n, 1 / n)
        self.trees_, self.alphas_ = [], []
        for t in range(self.n_estimators):
            tree = DecisionTreeClassifier(max_depth=self.max_depth,
                                          random_state=self.random_state + t)
            tree.fit(X, y, sample_weight=w)
            pred = tree.predict(X)
            err = np.sum(w[pred != y])
            if err >= 0.5:                      # worse than chance: stop
                if not self.trees_:             # keep at least one tree
                    self.trees_.append(tree); self.alphas_.append(1.0)
                break
            err = max(err, 1e-10)               # avoid log(inf) if err = 0
            alpha = 0.5 * np.log((1 - err) / err)
            self.trees_.append(tree); self.alphas_.append(alpha)
            if err <= 1e-10:                    # perfect tree: stop
                break
            w *= np.exp(-alpha * y * pred)      # reweight
            w /= w.sum()
        return self

    def decision_function(self, X):
        return sum(a * t.predict(X) for a, t in zip(self.alphas_, self.trees_))

    def predict(self, X):
        return np.where(self.decision_function(X) >= 0, 1, -1)


# ----------------------------------------------------------------- stacking
def default_base_models():
    return [DecisionTreeClassifier(max_depth=5, random_state=0),
            LogisticRegression(max_iter=1000),
            KNeighborsClassifier(n_neighbors=7)]


class MyStacking(BaseEstimator, ClassifierMixin):
    """Stacking: base models of different families, logistic-regression meta-model
    trained on out-of-fold probabilities of the class +1."""

    def __init__(self, base_models=None, n_folds=5, random_state=0):
        self.base_models = base_models
        self.n_folds = n_folds
        self.random_state = random_state

    def _meta_features(self, models, X):
        return np.column_stack([m.predict_proba(X)[:, 1] for m in models])

    def fit(self, X, y):
        self.classes_ = np.array([-1, 1])
        base = self.base_models or default_base_models()
        skf = StratifiedKFold(self.n_folds, shuffle=True,
                              random_state=self.random_state)
        Z = np.zeros((len(y), len(base)))   # out-of-fold predictions
        for tr, va in skf.split(X, y):
            for j, m in enumerate(base):
                Z[va, j] = clone(m).fit(X[tr], y[tr]).predict_proba(X[va])[:, 1]
        self.fitted_ = [clone(m).fit(X, y) for m in base]   # refit on all data
        self.meta_ = LogisticRegression(max_iter=1000).fit(Z, y)
        return self

    def decision_function(self, X):
        return self.meta_.decision_function(self._meta_features(self.fitted_, X))

    def predict(self, X):
        return self.meta_.predict(self._meta_features(self.fitted_, X))


# ------------------------------------------------------------ model catalog
def make_models():
    return {
        "Tree": DecisionTreeClassifier(random_state=0),
        "LogReg": LogisticRegression(max_iter=1000),
        "kNN": KNeighborsClassifier(n_neighbors=7),
        "Bagging": MyBagging(n_estimators=100),
        "RF": RandomForestClassifier(n_estimators=200, random_state=0, n_jobs=-1),
        "AdaBoost-d1": MyAdaBoost(n_estimators=100, max_depth=1),
        "AdaBoost-d3": MyAdaBoost(n_estimators=100, max_depth=3),
        "GBM": GradientBoostingClassifier(random_state=0),
        "HistGBM": HistGradientBoostingClassifier(random_state=0),
        "Stacking": MyStacking(),
    }


# ----------------------------------------------------------------- protocol
def get_scores(model, X):
    """Continuous score needed for the AUC."""
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    return model.predict_proba(X)[:, 1]


def compute_metrics(model, Xte, yte, minority_label):
    pred = model.predict(Xte)
    try:
        auc = roc_auc_score(yte, get_scores(model, Xte))
    except Exception:
        auc = np.nan  # model without scores
    return dict(acc=accuracy_score(yte, pred),
                bacc=balanced_accuracy_score(yte, pred),
                f1_min=f1_score(yte, pred, pos_label=minority_label),
                auc=auc)


def evaluate_model(model, X, y, n_rep=10, test_size=0.3, seed=0, noise=0.0):
    """Repeated stratified hold-out; same seed -> same splits for every model.
    Training labels are flipped with rate `noise`; test labels stay clean."""
    sss = StratifiedShuffleSplit(n_splits=n_rep, test_size=test_size,
                                 random_state=seed)
    out = []
    for r, (tr, te) in enumerate(sss.split(X, y)):
        scaler = StandardScaler().fit(X[tr])   # train statistics only
        Xtr, Xte = scaler.transform(X[tr]), scaler.transform(X[te])
        ytr = flip_labels(y[tr], noise, seed + r) if noise > 0 else y[tr]
        minority = 1 if np.mean(y[tr] == 1) < 0.5 else -1
        t0 = time.perf_counter()
        mdl = clone(model).fit(Xtr, ytr)
        t_fit = time.perf_counter() - t0
        out.append(dict(compute_metrics(mdl, Xte, y[te], minority), t_fit=t_fit))
    return pd.DataFrame(out)