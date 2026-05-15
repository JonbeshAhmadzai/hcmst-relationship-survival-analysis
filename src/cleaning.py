"""Cleaning helpers for the HCMST project.

These functions mirror the transformations used in
``notebooks/02_data_cleaning.ipynb`` while keeping the notebook itself as the
primary analysis artifact.
"""

from __future__ import annotations

import pandas as pd

from .utils import existing_columns, require_columns


CANDIDATE_VARIABLES = [
    "caseid_new",
    "w1_weight_combo",
    "w2_attrition_adj_weights",
    "w3_attrition_adj_weight",
    "w2_surveyed",
    "w3_surveyed",
    "w1_ppage",
    "w1_ppeduc",
    "w1_ppincimp",
    "w1_ppgender",
    "w1_ppethm",
    "w1_ppreg9",
    "w1_ppwork",
    "w1_partnership_status",
    "w1_q19",
    "w1_married",
    "w1_same_sex_couple",
    "w1_relate_duration_in2017_years",
    "w1_q34",
    "w1_q24_met_online",
    "w1_q24_met_through_friend",
    "w1_q24_met_through_family",
    "w1_q24_met_as_through_cowork",
    "w1_q24_school",
    "w1_q24_college",
    "w1_q24_church",
    "w1_q24_bar_restaurant",
    "w1_q24_party",
    "w1_q32",
    "w3_partner_type",
    "w3_live_w_partner",
    "w3_relationship_end_combo",
    "w3_partner_source",
    "w3_breakup_source",
    "w3_relationship_duration_yrs",
]

ACTIVE_RELATIONSHIP_STATUSES = ["married", "partnered, not married"]

ANALYSIS_VARIABLES = [
    "caseid_new",
    "w1_weight_combo",
    "w1_ppage",
    "w1_ppgender",
    "w1_ppeduc",
    "w1_ppincimp",
    "w1_ppethm",
    "w1_ppwork",
    "w1_partnership_status",
    "w1_q19",
    "w1_married",
    "w1_same_sex_couple_raw",
    "w1_relate_duration_in2017_years",
    "w1_q34",
    "w1_q24_met_online",
    "w2_surveyed",
    "w3_surveyed",
    "w2_attrition_adj_weights",
    "w3_attrition_adj_weight",
    "w3_partner_type",
    "w3_live_w_partner",
    "w3_relationship_end_combo",
    "w3_partner_source",
    "w3_breakup_source",
    "w3_relationship_duration_yrs",
]

ML_VARIABLES = [
    "caseid_new",
    "w1_weight_combo",
    "w1_ppage",
    "w1_ppgender_num",
    "w1_married",
    "w1_q24_met_online",
    "w2_surveyed",
    "w3_surveyed",
    "w1_same_sex_couple_num",
    "w1_relate_duration_in2017_years",
    "w1_q34_score",
    "w2_attrition_adj_weights",
    "w3_attrition_adj_weight",
]


def safe_numeric(series: pd.Series) -> pd.Series:
    """Convert a Series to numeric values, coercing invalid values to NaN."""

    return pd.to_numeric(series.astype(str), errors="coerce")


def recode_yes_no(series: pd.Series) -> pd.Series:
    """Recode yes/no strings to 1/0."""

    values = series.astype(str).str.strip().str.lower()
    return values.map({"yes": 1, "no": 0})


def recode_same_sex(series: pd.Series) -> pd.Series:
    """Recode HCMST same-sex couple labels to 1/0."""

    values = series.astype(str).str.strip()
    return values.map(
        {
            "same_sex_couple": 1,
            "NOT same-sex couple": 0,
            "NOT same-sex souple": 0,
        }
    )


def recode_quality(series: pd.Series) -> pd.Series:
    """Recode relationship quality labels to an ordinal 1-5 score."""

    values = series.astype(str).str.strip()
    return values.map(
        {
            "Very Poor": 1,
            "Poor": 2,
            "Fair": 3,
            "Good": 4,
            "Excellent": 5,
        }
    )


def select_candidate_variables(df: pd.DataFrame) -> pd.DataFrame:
    """Select the reduced variable set used in the cleaning notebook."""

    return df[existing_columns(df, CANDIDATE_VARIABLES)].copy()


def filter_wave1_relationship_sample(df: pd.DataFrame) -> pd.DataFrame:
    """Keep respondents in an active relationship at Wave 1."""

    require_columns(df, ["w1_partnership_status"], "selected variable dataset")
    mask = df["w1_partnership_status"].astype(str).isin(ACTIVE_RELATIONSHIP_STATUSES)
    return df.loc[mask].copy()


def clean_wave1_relationship_sample(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the Wave 1 recoding and imputation logic used by the project."""

    df_clean = df.copy()

    binary_cols = ["w1_married", "w1_q24_met_online", "w2_surveyed", "w3_surveyed"]
    for column in binary_cols:
        if column in df_clean.columns:
            df_clean[column] = recode_yes_no(df_clean[column])

    if "w1_same_sex_couple" in df_clean.columns:
        df_clean["w1_same_sex_couple_raw"] = df_clean["w1_same_sex_couple"].astype(str).str.strip()
        df_clean["w1_same_sex_couple"] = recode_same_sex(df_clean["w1_same_sex_couple"])
        df_clean["w1_same_sex_couple_num"] = recode_same_sex(df_clean["w1_same_sex_couple_raw"])

    if "w1_ppgender" in df_clean.columns:
        df_clean["w1_ppgender"] = df_clean["w1_ppgender"].astype(str).str.strip()
        df_clean["w1_ppgender_num"] = df_clean["w1_ppgender"].map({"Male": 0, "Female": 1})

    if "w1_q34" in df_clean.columns:
        df_clean["w1_q34_score"] = recode_quality(df_clean["w1_q34"])

    numeric_cols = [
        "w1_ppage",
        "w1_relate_duration_in2017_years",
        "w3_relationship_duration_yrs",
        "w1_weight_combo",
        "w2_attrition_adj_weights",
        "w3_attrition_adj_weight",
    ]
    for column in numeric_cols:
        if column in df_clean.columns:
            df_clean[column] = safe_numeric(df_clean[column])
            df_clean[column] = df_clean[column].fillna(df_clean[column].median())

    binary_impute_cols = [
        "w1_married",
        "w1_q24_met_online",
        "w2_surveyed",
        "w3_surveyed",
        "w1_same_sex_couple",
        "w1_same_sex_couple_num",
        "w1_ppgender_num",
    ]
    for column in binary_impute_cols:
        if column in df_clean.columns and df_clean[column].isna().any():
            df_clean[column] = df_clean[column].fillna(df_clean[column].mode()[0])

    if "w1_q34_score" in df_clean.columns:
        df_clean["w1_q34_score"] = df_clean["w1_q34_score"].fillna(df_clean["w1_q34_score"].median())

    text_cols = df_clean.select_dtypes(include=["object", "string", "category"]).columns
    for column in text_cols:
        df_clean[column] = df_clean[column].fillna("Missing")

    return df_clean


def make_analysis_ready(df_clean: pd.DataFrame) -> pd.DataFrame:
    """Create the readable analysis-ready dataset."""

    require_columns(df_clean, ANALYSIS_VARIABLES, "cleaned Wave 1 dataset")
    df_analysis = df_clean[ANALYSIS_VARIABLES].copy()
    text_cols = df_analysis.select_dtypes(include=["object", "string", "category"]).columns
    for column in text_cols:
        df_analysis[column] = df_analysis[column].astype(str).replace("nan", "Missing")
        df_analysis[column] = df_analysis[column].fillna("Missing")
    return df_analysis


def make_ml_ready(df_clean: pd.DataFrame) -> pd.DataFrame:
    """Create the numeric ML-ready dataset used for clustering."""

    require_columns(df_clean, ML_VARIABLES, "cleaned Wave 1 dataset")
    return df_clean[ML_VARIABLES].copy()


def build_cleaning_outputs(raw_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Run the full cleaning pipeline and return all main intermediate outputs."""

    selected = select_candidate_variables(raw_df)
    relationship_sample = filter_wave1_relationship_sample(selected)
    cleaned = clean_wave1_relationship_sample(relationship_sample)
    analysis_ready = make_analysis_ready(cleaned)
    ml_ready = make_ml_ready(cleaned)

    summary = pd.DataFrame(
        [
            {"dataset": "raw_full_dataset", "n_rows": len(raw_df), "n_columns": raw_df.shape[1]},
            {"dataset": "selected_variables_dataset", "n_rows": len(selected), "n_columns": selected.shape[1]},
            {"dataset": "wave1_relationship_sample", "n_rows": len(relationship_sample), "n_columns": relationship_sample.shape[1]},
            {"dataset": "wave1_cleaned", "n_rows": len(cleaned), "n_columns": cleaned.shape[1]},
            {"dataset": "wave1_analysis_ready", "n_rows": len(analysis_ready), "n_columns": analysis_ready.shape[1]},
            {"dataset": "wave1_ml_ready", "n_rows": len(ml_ready), "n_columns": ml_ready.shape[1]},
        ]
    )

    return {
        "selected_variables": selected,
        "wave1_relationship_sample": relationship_sample,
        "wave1_cleaned": cleaned,
        "wave1_analysis_ready": analysis_ready,
        "wave1_ml_ready": ml_ready,
        "cleaning_summary": summary,
    }
