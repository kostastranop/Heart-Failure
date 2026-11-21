"""Train one of the models chosen in Phase 3, score it and log the run to MLflow.

Usage:
    uv run python -m heart.train --model logreg
    uv run python -m heart.train --model random_forest
"""
import argparse

import mlflow
import pandas as pd
from matplotlib.figure import Figure
from mlflow.models import infer_signature

from heart import config as cfg
from heart.data import load_splits
from heart.evaluate import SCORING, cv_metrics, diagnostic_plots, test_metrics
from heart.models import TREE_PREP, build_tuned_model, get_specs

# Human-readable name of the preprocessing variant each model uses (see models.get_specs)
PREPROCESSING = {
    "default": "zeros->NaN + flag, IQR clip, median impute, Yeo-Johnson + robust scale, one-hot",
    "trees": "zeros->NaN + flag, median impute, one-hot (no clipping, no scaling)",
}

# MLflow saves sklearn models with skops, which (unlike pickle) only loads types it is told
# to trust. These are our own preprocessing code, a plain numpy type and sklearn's tree
# structure (used by the random forest), so they are safe.
SKOPS_TRUSTED_TYPES = [
    "heart.preprocessing.IQRClipper",
    "heart.preprocessing.zeros_to_nan",
    "numpy.dtype",
    "sklearn.tree._tree.Tree",
]


def train(name: str, X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series,
          y_test: pd.Series) -> tuple[object, dict[str, float], dict[str, Figure]]:
    """Cross-validate the tuned pipeline on train, refit it on all of train, score test once.

    Returns the fitted pipeline, the CV + test metrics, and the diagnostic plots.
    """
    model = build_tuned_model(name)

    metrics = cv_metrics(model, X_train, y_train)
    model.fit(X_train, y_train)
    metrics |= test_metrics(y_test, model.predict_proba(X_test)[:, 1])
    figures = diagnostic_plots(model, X_test, y_test, title=name)
    return model, metrics, figures


def print_report(name: str, metrics: dict[str, float]) -> None:
    """One line per metric: CV mean ± std next to the test score and its 95% interval."""
    print(f"\n{name}  (threshold {cfg.DECISION_THRESHOLD})")
    print(f"{'metric':<10} {'cv mean ± std':>15} {'test':>7} {'test 95% CI':>15}")
    for m in SCORING:
        print(f"{m:<10} {metrics[f'cv_{m}_mean']:>7.3f} ± {metrics[f'cv_{m}_std']:.3f}"
              f" {metrics[f'test_{m}']:>7.3f} "
              f"{metrics[f'test_{m}_ci_low']:>7.3f} – {metrics[f'test_{m}_ci_high']:.3f}")
    print(f"train - validation ROC AUC gap: {metrics['cv_roc_auc_gap']:.3f}")


def setup_mlflow() -> None:
    """Point MLflow at the local store and select (or create) the project experiment.

    The experiment is created with an explicit artifact folder, so artifacts land in
    mlruns/ at the project root wherever the script is started from.
    """
    mlflow.set_tracking_uri(cfg.MLFLOW_TRACKING_URI)
    if mlflow.get_experiment_by_name(cfg.EXPERIMENT_NAME) is None:
        mlflow.create_experiment(cfg.EXPERIMENT_NAME,
                                 artifact_location=cfg.MLFLOW_ARTIFACTS.as_uri())
    mlflow.set_experiment(cfg.EXPERIMENT_NAME)


def run_params(name: str, n_train: int, n_test: int) -> dict:
    """Everything needed to reproduce the run: tuned params, preprocessing, CV, threshold."""
    prep = "trees" if get_specs()[name][1] == TREE_PREP else "default"
    return {
        "model": name,
        **cfg.BEST_PARAMS[name],
        "preprocessing": PREPROCESSING[prep],
        "cv": f"RepeatedStratifiedKFold({cfg.CV_SPLITS} folds x {cfg.CV_REPEATS} repeats)",
        "decision_threshold": cfg.DECISION_THRESHOLD,
        "seed": cfg.SEED,
        "n_train": n_train,
        "n_test": n_test,
    }


def model_role(name: str) -> str:
    """'champion' or 'challenger' as chosen in Phase 3 (config.py); anything else is a candidate."""
    return {cfg.CHAMPION: "champion", cfg.CHALLENGER: "challenger"}.get(name, "candidate")


def log_run(name: str, model, metrics: dict[str, float], figures: dict[str, Figure],
            X_train: pd.DataFrame, y_train: pd.Series, n_test: int) -> tuple[str, int]:
    """Log one trained model to MLflow, register it, and return (run id, model version).

    The logged model is the whole pipeline (preprocessing + estimator), so whoever loads it
    sends raw patient data and gets a prediction, with no preprocessing to redo by hand.
    """
    role = model_role(name)
    with mlflow.start_run(run_name=name) as run:
        mlflow.set_tags({"phase": "4", "role": role})
        mlflow.log_params(run_params(name, len(X_train), n_test))
        mlflow.log_metrics(metrics)
        for fname, fig in figures.items():
            mlflow.log_figure(fig, f"plots/{fname}")

        # Record which data the model was trained on (shown in the run's Datasets panel)
        train_df = X_train.assign(**{cfg.TARGET: y_train})
        mlflow.log_input(mlflow.data.from_pandas(train_df, source=str(cfg.TRAIN_CSV),
                                                 targets=cfg.TARGET, name="train"),
                         context="training")

        # Signature = the input columns/types and the output the model accepts and returns;
        # MLflow checks incoming data against it when the model is served.
        example = X_train.head(5)
        info = mlflow.sklearn.log_model(
            model,
            name="model",
            signature=infer_signature(X_train, model.predict(X_train)),
            input_example=example,
            # The pipeline calls functions from the heart package (e.g. zeros_to_nan),
            # so the package is stored with the model to load it anywhere, e.g. in Docker.
            code_paths=[str(cfg.ROOT / "src" / "heart")],
            skops_trusted_types=SKOPS_TRUSTED_TYPES,
            # Each training run adds a new version of the registered model
            registered_model_name=cfg.REGISTERED_MODEL,
        )

    version = int(info.registered_model_version)
    set_alias(name, version, role)
    return run.info.run_id, version


def set_alias(name: str, version: int, role: str) -> None:
    """Tag the new version with its model name and point the role alias at it.

    The app loads the model through the alias (models:/heart-disease@champion), never a
    version number, so promoting a retrained model only means moving the alias.
    """
    client = mlflow.MlflowClient()
    client.set_model_version_tag(cfg.REGISTERED_MODEL, str(version), "model", name)
    if role != "candidate":
        client.set_registered_model_alias(cfg.REGISTERED_MODEL, role, version)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=list(cfg.BEST_PARAMS), default=cfg.CHAMPION)
    args = parser.parse_args()

    setup_mlflow()
    X_train, X_test, y_train, y_test = load_splits()
    model, metrics, figures = train(args.model, X_train, X_test, y_train, y_test)
    print_report(args.model, metrics)

    run_id, version = log_run(args.model, model, metrics, figures, X_train, y_train,
                              len(X_test))
    print(f"\nlogged to MLflow: experiment '{cfg.EXPERIMENT_NAME}', run {run_id}")
    print(f"registered: {cfg.REGISTERED_MODEL} v{version} (alias: {model_role(args.model)})")


if __name__ == "__main__":
    main()
