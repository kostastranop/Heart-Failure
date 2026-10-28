import pytest

from heart import config as cfg
from heart.data import load_raw, make_splits
from heart.models import TREE_PREP
from heart.preprocessing import build_preprocessor

# Numeric + one 0/1 column per binary + one column per nominal level + missing flag
N_FEATURES_OUT = (
    len(cfg.NUM_COLS)
    + len(cfg.BINARY_COLS)
    + sum(len(cfg.CATEGORY_LEVELS[c]) for c in cfg.NOMINAL_COLS)
    + 1
)


@pytest.fixture(scope="module")
def splits():
    return make_splits(load_raw())


# Both preprocessing variants used in modeling: default (linear/kNN) and trees
@pytest.mark.parametrize("prep", [{}, TREE_PREP], ids=["default", "trees"])
def test_fit_on_train_transform_test(splits, prep):
    X_train, X_test, _, _ = splits
    preprocessor = build_preprocessor(**prep).fit(X_train)

    Xt_train = preprocessor.transform(X_train)
    Xt_test = preprocessor.transform(X_test)

    assert Xt_train.shape == (len(X_train), N_FEATURES_OUT)
    assert Xt_test.shape == (len(X_test), N_FEATURES_OUT)
    assert list(Xt_test.columns) == list(Xt_train.columns)
    assert not Xt_train.isna().any().any()
    assert not Xt_test.isna().any().any()
