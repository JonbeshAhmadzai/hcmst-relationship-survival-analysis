"""Modeling helpers for Wave 3 relationship survival."""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .utils import require_columns


TARGET = "still_together_w3"
WEIGHT_COL = "w1_weight_combo"

MEETING_VARS = ["w1_q24_met_online"]

DEMO_VARS = [
    "w1_ppage",
    "w1_ppgender",
    "w1_ppeduc",
    "w1_ppincimp",
    "w1_ppethm",
    "w1_ppwork",
]

STRUCTURE_VARS = [
    "w1_married",
    "w1_same_sex_couple_raw",
    "w1_relate_duration_in2017_years",
]

QUALITY_VARS = ["w1_q34"]

MODEL_SPECS = {
    "Model 1: meeting_online_only": MEETING_VARS,
    "Model 2: + demographics": MEETING_VARS + DEMO_VARS,
    "Model 3: + relationship_structure": MEETING_VARS + DEMO_VARS + STRUCTURE_VARS,
    "Model 4: + relationship_quality": MEETING_VARS + DEMO_VARS + STRUCTURE_VARS + QUALITY_VARS,
}


def add_wave3_survival_target(df: pd.DataFrame, target_col: str = TARGET) -> pd.DataFrame:
    """Create the Wave 3 relationship survival target used in the project."""

    require_columns(
        df,
        ["w3_relationship_end_combo", "w3_partner_type"],
        "analysis-ready dataset",
    )
    modeled = df.copy()
    modeled[target_col] = pd.NA

    survived_mask = (
        (modeled["w3_relationship_end_combo"] == "no report of breakup or partner death")
        | modeled["w3_partner_type"].isin(["married", "in unmarried partnership"])
    )
    ended_mask = (
        (modeled["w3_partner_type"] == "unpartnered")
        | modeled["w3_relationship_end_combo"].isin(["Separated/ Broke up", "Got Divorced"])
    )

    modeled.loc[survived_mask, target_col] = 1
    modeled.loc[ended_mask, target_col] = 0
    modeled[target_col] = pd.to_numeric(modeled[target_col], errors="coerce")
    return modeled


def make_modeling_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Create the compact modeling dataset from the analysis-ready data."""

    modeled = add_wave3_survival_target(df)
    features = MEETING_VARS + DEMO_VARS + STRUCTURE_VARS + QUALITY_VARS
    required = features + [TARGET, WEIGHT_COL]
    require_columns(modeled, required, "target-enriched dataset")
    return modeled.dropna(subset=[TARGET])[required].copy()


def split_features_target(
    model_data: pd.DataFrame,
    target_col: str = TARGET,
    weight_col: str = WEIGHT_COL,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Split model data into features, target, and sample weights."""

    require_columns(model_data, [target_col, weight_col], "modeling dataset")
    X = model_data.drop(columns=[target_col, weight_col]).copy()
    y = model_data[target_col].copy()
    weights = model_data[weight_col].copy()
    return X, y, weights


def infer_feature_types(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Infer numeric and categorical predictors for preprocessing."""

    numeric_features = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_features = [column for column in X.columns if column not in numeric_features]
    return numeric_features, categorical_features


def make_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Build the preprocessing pipeline used before logistic regression."""

    numeric_features, categorical_features = infer_feature_types(X)

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ]
    )


def make_logistic_pipeline(X: pd.DataFrame, random_state: int = 42) -> Pipeline:
    """Create a logistic-regression modeling pipeline."""

    return Pipeline(
        steps=[
            ("preprocess", make_preprocessor(X)),
            (
                "model",
                LogisticRegression(max_iter=1000, random_state=random_state),
            ),
        ]
    )


def train_test_split_with_weights(
    X: pd.DataFrame,
    y: pd.Series,
    weights: pd.Series,
    test_size: float = 0.25,
    random_state: int = 42,
):
    """Create a stratified train-test split including sample weights."""

    return train_test_split(
        X,
        y,
        weights,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )


def evaluate_predictions(y_true: pd.Series, y_pred: pd.Series, y_proba: pd.Series | None = None) -> dict[str, float]:
    """Calculate the classification metrics reported in the project."""

    results = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    if y_proba is not None:
        results["roc_auc"] = roc_auc_score(y_true, y_proba)
    return results


def evaluate_model_specs(
    model_data: pd.DataFrame,
    model_specs: dict[str, list[str]] = MODEL_SPECS,
    test_size: float = 0.25,
    random_state: int = 42,
) -> pd.DataFrame:
    """Train and evaluate the project's logistic-regression specifications."""

    rows = []
    for model_name, features in model_specs.items():
        require_columns(model_data, features + [TARGET, WEIGHT_COL], model_name)
        subset = model_data[features + [TARGET, WEIGHT_COL]].copy()
        X, y, weights = split_features_target(subset)
        X_train, X_test, y_train, y_test, weights_train, _ = train_test_split_with_weights(
            X,
            y,
            weights,
            test_size=test_size,
            random_state=random_state,
        )

        pipeline = make_logistic_pipeline(X_train, random_state=random_state)
        pipeline.fit(X_train, y_train, model__sample_weight=weights_train)

        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]
        metrics = evaluate_predictions(y_test, y_pred, y_proba)
        rows.append({"model": model_name, **{key: round(value, 3) for key, value in metrics.items()}})

    majority_class = y.mode()[0]
    baseline_pred = pd.Series([majority_class] * len(y_test), index=y_test.index)
    baseline_metrics = evaluate_predictions(y_test, baseline_pred, pd.Series([0.5] * len(y_test), index=y_test.index))
    rows.append(
        {
            "model": "Naive majority baseline",
            **{key: round(value, 3) for key, value in baseline_metrics.items()},
        }
    )

    return pd.DataFrame(rows)


def extract_logistic_coefficients(pipeline: Pipeline) -> pd.DataFrame:
    """Extract sorted coefficients from a fitted logistic-regression pipeline."""

    preprocessor = pipeline.named_steps["preprocess"]
    model = pipeline.named_steps["model"]
    feature_names = preprocessor.get_feature_names_out()
    coef_df = pd.DataFrame(
        {
            "feature": feature_names,
            "coefficient": model.coef_[0],
        }
    )
    coef_df["abs_coefficient"] = coef_df["coefficient"].abs()
    return coef_df.sort_values("abs_coefficient", ascending=False).reset_index(drop=True)
