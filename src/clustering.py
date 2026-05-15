"""Clustering helpers for the HCMST project."""

from __future__ import annotations

import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from .utils import require_columns


CLUSTER_FEATURES_WITH_GENDER = [
    "w1_ppage",
    "w1_ppgender_num",
    "w1_married",
    "w1_q24_met_online",
    "w1_same_sex_couple_num",
    "w1_relate_duration_in2017_years",
    "w1_q34_score",
]

CLUSTER_FEATURES_NO_GENDER = [
    "w1_ppage",
    "w1_married",
    "w1_q24_met_online",
    "w1_same_sex_couple_num",
    "w1_relate_duration_in2017_years",
    "w1_q34_score",
]


def scale_features(df: pd.DataFrame, features: list[str]) -> tuple[pd.DataFrame, StandardScaler]:
    """Standardize clustering features and return a DataFrame plus scaler."""

    require_columns(df, features, "clustering dataset")
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df[features])
    scaled_df = pd.DataFrame(scaled, columns=features, index=df.index)
    return scaled_df, scaler


def evaluate_kmeans_range(
    df: pd.DataFrame,
    features: list[str],
    k_values: range | list[int] = range(2, 11),
    random_state: int = 42,
    n_init: int = 10,
) -> pd.DataFrame:
    """Calculate inertia and silhouette scores for a range of K values."""

    scaled_df, _ = scale_features(df, features)
    rows = []
    for k in k_values:
        model = KMeans(n_clusters=k, random_state=random_state, n_init=n_init)
        labels = model.fit_predict(scaled_df)
        rows.append(
            {
                "k": k,
                "inertia": model.inertia_,
                "silhouette": silhouette_score(scaled_df, labels),
            }
        )
    return pd.DataFrame(rows)


def fit_kmeans_clusters(
    df: pd.DataFrame,
    features: list[str],
    n_clusters: int = 6,
    label_col: str = "cluster",
    random_state: int = 42,
    n_init: int = 10,
) -> tuple[pd.DataFrame, KMeans, StandardScaler]:
    """Fit K-Means and return a copy of the data with cluster labels."""

    scaled_df, scaler = scale_features(df, features)
    model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=n_init)
    clustered = df.copy()
    clustered[label_col] = model.fit_predict(scaled_df)
    return clustered, model, scaler


def make_cluster_profile(
    df: pd.DataFrame,
    features: list[str],
    label_col: str,
    round_digits: int = 2,
) -> pd.DataFrame:
    """Summarize cluster means and cluster size percentages."""

    require_columns(df, [label_col] + features, "clustered dataset")
    profile = df.groupby(label_col)[features].mean().round(round_digits)
    sizes = df[label_col].value_counts(normalize=True).sort_index().mul(100).round(1)
    profile["cluster_size_%"] = sizes
    return profile.reset_index()


def fit_hierarchical_clusters(
    df: pd.DataFrame,
    features: list[str] = CLUSTER_FEATURES_NO_GENDER,
    n_clusters: int = 6,
    label_col: str = "cluster_h",
    method: str = "ward",
) -> tuple[pd.DataFrame, object, StandardScaler]:
    """Fit hierarchical clustering and return labels plus linkage matrix."""

    scaled_df, scaler = scale_features(df, features)
    linkage_matrix = linkage(scaled_df, method=method)
    clustered = df.copy()
    clustered[label_col] = fcluster(linkage_matrix, n_clusters, criterion="maxclust")
    return clustered, linkage_matrix, scaler
