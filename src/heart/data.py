import pandas as pd
import great_expectations as gx
import great_expectations.expectations as gxe
from sklearn.model_selection import train_test_split
 
from heart import config as cfg

# Data validation function with great expectations
def validate(df):
    # Initialize context
    context = gx.get_context(mode="ephemeral")

    # Register pandas DataFrame as a batch
    data_source = context.data_sources.add_pandas("heart_failure_source")
    data_asset = data_source.add_dataframe_asset(name="heart_failure_asset")
    batch_definition = data_asset.add_batch_definition_whole_dataframe("heart_failure_batch")
    batch = batch_definition.get_batch(batch_parameters={"dataframe": df})

    # Create Expectation Suite
    suite = context.suites.add(gx.ExpectationSuite(name="heart_failure_validation_suite"))

    # Schema validation
    suite.add_expectation(gxe.ExpectColumnToExist(column=cfg.TARGET))
    suite.add_expectation(gxe.ExpectColumnValuesToNotBeNull(column=cfg.TARGET))
    for col in cfg.FEATURE_COLS:
        suite.add_expectation(gxe.ExpectColumnToExist(column=col))
        suite.add_expectation(gxe.ExpectColumnValuesToNotBeNull(column=col))

    # Categorical values validation
    suite.add_expectation(gxe.ExpectColumnValuesToBeInSet(column="Sex", value_set=["M", "F"]))
    suite.add_expectation(gxe.ExpectColumnValuesToBeInSet(column="ChestPainType", value_set=["TA", "ATA", "NAP", "ASY"]))
    suite.add_expectation(gxe.ExpectColumnValuesToBeInSet(column="RestingECG", value_set=["Normal", "ST", "LVH"]))
    suite.add_expectation(gxe.ExpectColumnValuesToBeInSet(column="ExerciseAngina", value_set=["Y", "N"]))
    suite.add_expectation(gxe.ExpectColumnValuesToBeInSet(column="ST_Slope", value_set=["Up", "Flat", "Down"]))

    # Numerical values validation
    suite.add_expectation(gxe.ExpectColumnValuesToBeBetween(column="Age", min_value=20, max_value=90))
    suite.add_expectation(gxe.ExpectColumnValuesToBeBetween(column="RestingBP", min_value=0, max_value=200))
    suite.add_expectation(gxe.ExpectColumnValuesToBeBetween(column="Cholesterol", min_value=0, max_value=610))
    suite.add_expectation(gxe.ExpectColumnValuesToBeBetween(column="FastingBS", min_value=0, max_value=1))
    suite.add_expectation(gxe.ExpectColumnValuesToBeBetween(column="MaxHR", min_value=60, max_value=202))
    suite.add_expectation(gxe.ExpectColumnValuesToBeBetween(column="Oldpeak", min_value=-3, max_value=7))
    suite.add_expectation(gxe.ExpectColumnValuesToBeBetween(column="HeartDisease", min_value=0, max_value=1))

    # Results & processing
    results = batch.validate(suite)

    failed_expectations = []
    for r in results.results:
        if not r.success:
            failed_expectations.append(r.expectation_config.type)

    total_checks = len(results.results)
    passed_checks = sum(1 for r in results.results if r.success)
    failed_checks = total_checks - passed_checks

    if results.success:
        print(f"Data validation PASSED: {passed_checks}/{total_checks} checks successful")
    else:
        print(f"Data validation FAILED: {failed_checks}/{total_checks} checks failed")
        print(f"Failed expectations: {failed_expectations}")

    return results.success, failed_expectations

# Data loading function
def load_raw(path=cfg.RAW_PATH):
    df = pd.read_csv(path)
    validate(df)
    return df

# Data splitting function
def make_splits(df, test_size=cfg.TEST_SIZE, seed=cfg.SEED):
    X = df[cfg.FEATURE_COLS]
    y = df[cfg.TARGET]

    # Create a stratification variable that combines the target and whether cholesterol is zero
    strata = y.astype(str) + "_" + (df["Cholesterol"] == 0).astype(str)
 
    return train_test_split(
        X, y, test_size=test_size, stratify=strata, random_state=seed
    )
