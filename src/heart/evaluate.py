"""Metrics and diagnostic plots for train.py.

The metric set, CV scheme, bootstrap and plots are the same as in 02_modeling.ipynb, so
what is logged to MLflow can be checked against the notebook.
"""
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate

from heart import config as cfg

# sklearn scorer names; brier is negated by sklearn ("higher is better"), flipped back below
SCORING = {
    "roc_auc": "roc_auc",
    "pr_auc": "average_precision",
    "brier": "neg_brier_score",
    "recall": "recall",
    "precision": "precision",
    "f1": "f1",
    "accuracy": "accuracy",
}


def cv_metrics(model, X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
    """Repeated stratified 10-fold CV on the training set.

    Returns mean and std of every metric across the 50 folds, plus the train-validation
    ROC AUC gap as an overfitting check. The pipeline is refit in every fold, so the
    preprocessing never sees the validation rows.
    """
    cv = RepeatedStratifiedKFold(n_splits=cfg.CV_SPLITS, n_repeats=cfg.CV_REPEATS,
                                 random_state=cfg.SEED)
    raw = cross_validate(model, X, y, cv=cv, scoring=SCORING,
                         return_train_score=True, n_jobs=-1)
    out = {}
    for m in SCORING:
        scores = -raw[f"test_{m}"] if m == "brier" else raw[f"test_{m}"]
        out[f"cv_{m}_mean"] = float(np.mean(scores))
        out[f"cv_{m}_std"] = float(np.std(scores, ddof=1))
    out["cv_roc_auc_gap"] = float(np.mean(raw["train_roc_auc"]) - out["cv_roc_auc_mean"])
    return out


def score(y: np.ndarray, proba: np.ndarray,
          threshold: float = cfg.DECISION_THRESHOLD) -> dict[str, float]:
    """All metrics for one set of true labels and predicted disease probabilities."""
    pred = (proba >= threshold).astype(int)
    return {
        "roc_auc": roc_auc_score(y, proba),
        "pr_auc": average_precision_score(y, proba),
        "brier": brier_score_loss(y, proba),
        "recall": recall_score(y, pred),
        "precision": precision_score(y, pred),
        "f1": f1_score(y, pred),
        "accuracy": accuracy_score(y, pred),
    }


def test_metrics(y: pd.Series, proba: np.ndarray) -> dict[str, float]:
    """Score the held-out test set once, with a bootstrap 95% interval per metric.

    The test set has only 184 patients, so a single number can mislead; resampling the
    patients with replacement shows how much each score could move.
    """
    y = np.asarray(y)
    rng = np.random.default_rng(cfg.SEED)
    boot = []
    for _ in range(cfg.N_BOOTSTRAP):
        idx = rng.integers(0, len(y), len(y))
        boot.append(score(y[idx], proba[idx]))
    boot = pd.DataFrame(boot)

    out = {}
    for m, value in score(y, proba).items():
        out[f"test_{m}"] = float(value)
        out[f"test_{m}_ci_low"] = float(boot[m].quantile(0.025))
        out[f"test_{m}_ci_high"] = float(boot[m].quantile(0.975))
    return out


# Plots
# Same palette as 02_modeling.ipynb
BLUE, ORANGE, GRID, INK = "#2a78d6", "#eb6834", "#d9d8d3", "#52514e"
STYLE = {"color": INK, "fontsize": 10, "loc": "left"}


def _new_figure(width: float, height: float, ncols: int = 1) -> tuple[Figure, list]:
    """Figure built without pyplot, so the script needs no display and leaks no figures."""
    fig = Figure(figsize=(width, height), dpi=150, layout="tight")
    axes = [fig.add_subplot(1, ncols, i + 1) for i in range(ncols)]
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color(GRID)
        ax.tick_params(colors=INK)
    return fig, axes


def plot_confusion_matrix(y: pd.Series, proba: np.ndarray, title: str) -> Figure:
    """Confusion matrix at the decision threshold.

    Bottom-left (false negatives = sick patients sent home) is the cell that matters most.
    """
    fig, (ax,) = _new_figure(4.2, 3.6)
    pred = (proba >= cfg.DECISION_THRESHOLD).astype(int)
    ConfusionMatrixDisplay.from_predictions(y, pred, display_labels=["no disease", "disease"],
                                            cmap="Blues", colorbar=False, ax=ax)
    ax.set_title(f"{title} (threshold {cfg.DECISION_THRESHOLD})", **STYLE)
    return fig


def plot_roc_pr(y: pd.Series, proba: np.ndarray, title: str) -> Figure:
    """ROC curve (dot = decision threshold) and precision-recall curve side by side.

    Dashed lines show a random model: the diagonal for ROC, the disease rate for PR.
    """
    fig, (ax_roc, ax_pr) = _new_figure(10, 4, ncols=2)
    RocCurveDisplay.from_predictions(y, proba, name=title, curve_kwargs={"color": BLUE},
                                     ax=ax_roc)
    tn, fp, fn, tp = confusion_matrix(y, proba >= cfg.DECISION_THRESHOLD).ravel()
    ax_roc.scatter(fp / (fp + tn), tp / (tp + fn), color=ORANGE, s=40, zorder=3,
                   label=f"threshold {cfg.DECISION_THRESHOLD}")
    ax_roc.plot([0, 1], [0, 1], color=GRID, linestyle="--", linewidth=1)
    ax_roc.set_title("ROC curve", **STYLE)
    ax_roc.legend(frameon=False, loc="lower right")

    PrecisionRecallDisplay.from_predictions(y, proba, name=title,
                                            curve_kwargs={"color": BLUE}, ax=ax_pr)
    ax_pr.axhline(np.mean(y), color=GRID, linestyle="--", linewidth=1)
    ax_pr.set_title("Precision-recall curve", **STYLE)
    ax_pr.legend(frameon=False, loc="lower left")
    return fig


def plot_permutation_importance(model, X: pd.DataFrame, y: pd.Series, title: str) -> Figure:
    """Drop in test ROC AUC when one raw feature is shuffled, with std over the shuffles.

    Works on the whole pipeline, so a one-hot encoded feature counts as one feature and
    any model can be compared with any other. Nothing is fitted, the model only predicts.
    """
    r = permutation_importance(model, X, y, scoring="roc_auc",
                               n_repeats=cfg.N_PERMUTATION_REPEATS,
                               random_state=cfg.SEED, n_jobs=-1)
    imp = pd.DataFrame({"mean": r.importances_mean, "std": r.importances_std},
                       index=X.columns).sort_values("mean")

    fig, (ax,) = _new_figure(6, 4.2)
    ax.barh(imp.index, imp["mean"], xerr=imp["std"], color=BLUE,
            error_kw={"ecolor": INK, "linewidth": 0.8})
    ax.axvline(0, color=INK, linewidth=0.8)
    ax.set_xlabel("drop in test ROC AUC when shuffled", color=INK)
    ax.set_title(f"{title}: permutation importance (test set)", **STYLE)
    return fig


def diagnostic_plots(model, X_test: pd.DataFrame, y_test: pd.Series,
                     title: str) -> dict[str, Figure]:
    """All diagnostic plots of a fitted model on the test set, keyed by file name."""
    proba = model.predict_proba(X_test)[:, 1]
    return {
        "confusion_matrix.png": plot_confusion_matrix(y_test, proba, title),
        "roc_pr_curves.png": plot_roc_pr(y_test, proba, title),
        "permutation_importance.png": plot_permutation_importance(model, X_test, y_test,
                                                                  title),
    }
