import mlflow
import numpy as np
import pytest

from heart import config as cfg
from heart import train as train_script
from heart.data import load_raw, make_splits
from heart.evaluate import score
from heart.models import build_tuned_model


@pytest.fixture(scope="module")
def splits():
    return make_splits(load_raw())


# The training script must reproduce the tuned models, so it has to use config.BEST_PARAMS
@pytest.mark.parametrize("name", list(cfg.BEST_PARAMS))
def test_tuned_model_uses_config_params(name):
    estimator = build_tuned_model(name)[-1]
    params = estimator.get_params()
    assert all(params[k] == v for k, v in cfg.BEST_PARAMS[name].items())


# End to end: log + register the champion in a throwaway MLflow store, then load it the way
# the API will (by alias) and check it predicts exactly like the model that was trained.
def test_champion_registered_and_loadable_by_alias(splits, tmp_path, monkeypatch):
    X_train, X_test, y_train, y_test = splits
    monkeypatch.setattr(cfg, "MLFLOW_TRACKING_URI", f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}")
    monkeypatch.setattr(cfg, "MLFLOW_ARTIFACTS", tmp_path / "mlruns")
    train_script.setup_mlflow()

    model = build_tuned_model(cfg.CHAMPION).fit(X_train, y_train)
    # Test-set scores only: the 50-fold CV and the plots are not needed to test the logging
    metrics = score(y_test, model.predict_proba(X_test)[:, 1])
    _, version = train_script.log_run(cfg.CHAMPION, model, metrics, {}, X_train, y_train,
                                      len(X_test))

    loaded = mlflow.sklearn.load_model(f"models:/{cfg.REGISTERED_MODEL}@champion")
    assert version == 1
    np.testing.assert_allclose(loaded.predict_proba(X_test), model.predict_proba(X_test))
