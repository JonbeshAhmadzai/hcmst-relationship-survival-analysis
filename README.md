# HCMST Relationship Survival Analysis

Portfolio data science project using the **How Couples Meet and Stay Together (HCMST) 2017-2022** longitudinal survey to study how couples meet, how relationship profiles differ, and which baseline characteristics predict whether couples remain together over time.

The project combines exploratory analysis, data cleaning, unsupervised clustering, and supervised modeling to answer a practical question relevant to online dating platforms:

> Does meeting online predict long-term relationship survival, or do relationship quality and structure matter more?

## Project Overview

The HCMST dataset follows respondents across three waves:

- **Wave 1:** 2017 baseline survey
- **Wave 2:** 2020 follow-up
- **Wave 3:** 2022 follow-up

This analysis focuses on respondents who were in an active relationship at Wave 1, then uses later-wave information to evaluate whether those relationships were still intact by Wave 3.

The project has two main analytical tracks:

1. **Relationship segmentation**
   - Use clustering to identify meaningful relationship profiles among Wave 1 partnered respondents.

2. **Relationship survival modeling**
   - Use Wave 1 baseline characteristics to predict whether a relationship survived to Wave 3.

## Research Questions

The project investigates:

1. What relationship profiles exist among couples in the Wave 1 active-relationship sample?
2. Does meeting online in Wave 1 predict whether couples are still together in Wave 3?
3. Does baseline relationship quality predict long-term relationship survival better than meeting context?
4. What insights could an online dating platform such as OKCupid use from these findings?

## Dataset

The project uses the public HCMST 2017-2022 dataset:

- Raw dataset size: **3,510 respondents**
- Raw variables: **725 columns**
- Wave 1 active relationship sample: **2,862 respondents**
- Known Wave 3 modeling sample: **1,397 respondents**

Key variables include:

- respondent demographics
- relationship status
- marital status
- same-sex couple indicator
- relationship duration
- relationship quality
- whether the couple met online
- Wave 3 partner status and relationship-end information

Raw data and documentation are stored in:

```text
data/raw/
```

Processed datasets are stored in:

```text
data/processed/
```

## Repository Structure

```text
hcmst_project/
|-- data/
|   |-- raw/
|   |   |-- HCMST 2017 to 2022 small public version 2.2.sav
|   |   |-- HCMST 2017 to 2022 v2.2 codebook.pdf
|   |   `-- HCMST 2017- 2022 user's guide v2.3.pdf
|   `-- processed/
|       |-- hcmst_selected_variables.csv
|       |-- hcmst_wave1_relationship_sample.csv
|       |-- hcmst_wave1_cleaned.csv
|       |-- hcmst_wave1_analysis_ready.csv
|       |-- hcmst_wave1_ml_ready.csv
|       |-- hcmst_wave1_ml_with_clusters.csv
|       `-- hcmst_wave3_modeling_dataset.csv
|-- notebooks/
|   |-- 01_data_exploration.ipynb
|   |-- 02_data_cleaning.ipynb
|   |-- 03_clustering.ipynb
|   |-- 04_research_question_model.ipynb
|   `-- 05_final_figures.ipynb
|-- outputs/
|   |-- figures/
|   `-- tables/
|-- src/
|   |-- cleaning.py
|   |-- clustering.py
|   |-- modeling.py
|   `-- utils.py
|-- requirements.txt
`-- README.md
```

Note: the main analysis remains notebook-driven, while `src/` contains reusable helper functions that mirror the notebook workflow for cleaning, clustering, modeling, and shared project utilities.

## Methodology

### 1. Exploratory Data Analysis

Notebook: `notebooks/01_data_exploration.ipynb`

The first notebook validates the dataset structure, reviews wave-specific variables, and summarizes the main demographic, relationship, meeting-context, and Wave 3 outcome variables.

Key outputs:

- `outputs/tables/initial_variable_dictionary.csv`
- `outputs/tables/wave1_demographics_summary.csv`
- `outputs/tables/wave1_relationship_summary.csv`
- `outputs/tables/wave1_how_couples_met_summary.csv`
- `outputs/tables/wave3_outcome_candidates_summary.csv`

### 2. Data Cleaning

Notebook: `notebooks/02_data_cleaning.ipynb`

The cleaning stage creates a smaller working dataset, filters to respondents in active Wave 1 relationships, recodes analysis variables, handles missing values, and saves both readable and machine-learning-ready datasets.

Final cleaning outputs:

| Dataset | Rows | Columns | Purpose |
|---|---:|---:|---|
| `hcmst_selected_variables.csv` | 3,510 | 35 | Reduced variable set |
| `hcmst_wave1_relationship_sample.csv` | 2,862 | 35 | Active relationship sample |
| `hcmst_wave1_cleaned.csv` | 2,862 | 39 | Cleaned working file |
| `hcmst_wave1_analysis_ready.csv` | 2,862 | 25 | Interpretable analysis file |
| `hcmst_wave1_ml_ready.csv` | 2,862 | 13 | Numeric ML-ready file |

### 3. Clustering

Notebook: `notebooks/03_clustering.ipynb`

The clustering analysis identifies relationship profiles using Wave 1 baseline characteristics.

Models used:

- K-Means clustering with gender
- K-Means clustering without gender
- Hierarchical clustering with Ward linkage

The preferred specification is **K-Means without gender**, because the first K-Means model showed that respondent gender was driving some cluster separation. Removing gender produced more substantively meaningful relationship profiles.

Preferred clustering features:

- respondent age
- marital status
- met online
- same-sex couple indicator
- relationship duration
- relationship quality

Preferred K-Means cluster profiles:

| Cluster | Profile | Size |
|---:|---|---:|
| 0 | Online-meeting couples | 8.9% |
| 1 | Midlife married couples | 28.0% |
| 2 | Younger unmarried couples | 16.2% |
| 3 | Same-sex couples | 7.6% |
| 4 | Lower-quality older/midlife married couples | 5.6% |
| 5 | Older long-term married couples | 33.6% |

Key clustering outputs:

- `outputs/tables/cluster_profile_kmeans_without_gender.csv`
- `outputs/tables/cluster_profile_hierarchical.csv`
- `outputs/figures/final_cluster_profile_heatmap.png`
- `outputs/figures/final_cluster_sizes.png`
- `outputs/figures/final_kmeans_vs_hierarchical_sizes.png`

### 4. Relationship Survival Modeling

Notebook: `notebooks/04_research_question_model.ipynb`

The modeling stage predicts whether a Wave 1 relationship was still intact by Wave 3.

Target variable:

- `1`: relationship survived to Wave 3
- `0`: relationship ended by Wave 3

Known Wave 3 modeling sample:

| Outcome | Count |
|---|---:|
| Still together | 1,250 |
| Relationship ended | 147 |

Because the target is highly imbalanced, model performance is interpreted mainly using **ROC AUC**, not accuracy.

Models compared:

| Model | Predictors | ROC AUC |
|---|---|---:|
| Naive majority baseline | Predicts survival for everyone | 0.500 |
| Model 1 | Meeting context only | 0.631 |
| Model 2 | Meeting context + demographics | 0.513 |
| Model 3 | Meeting context + demographics + relationship structure | 0.658 |
| Model 4 | Meeting context + demographics + relationship structure + relationship quality | 0.678 |

The best model included relationship quality, relationship structure, demographics, and meeting context.

Key modeling outputs:

- `outputs/tables/model_comparison_results.csv`
- `outputs/tables/logistic_regression_coefficients.csv`
- `outputs/figures/final_model_comparison_roc_auc.png`
- `outputs/figures/final_top_coefficients.png`

## Key Findings

### Relationship Profiles Are Meaningful

The clustering analysis consistently identified several broad relationship types:

- older long-term married couples
- midlife married couples
- younger unmarried couples
- online-meeting couples
- same-sex couples
- lower-quality established marriages

These segments were not random mathematical artifacts. They reflected meaningful differences in age, marital status, relationship duration, meeting context, same-sex relationship status, and relationship quality.

### Meeting Online Has Predictive Value, But It Is Modest

The model using only meeting context achieved:

```text
ROC AUC = 0.631
```

This is better than random chance, meaning online/offline meeting context contains some information about long-term relationship survival. However, it is not a strong standalone predictor.

### Relationship Quality Matters More Than Meeting Context

The strongest model achieved:

```text
ROC AUC = 0.678
```

This model included relationship quality, which improved performance beyond meeting context alone. The result suggests that how respondents evaluated their relationship at baseline matters more than whether the couple met online or offline.

### Accuracy Is Misleading

Approximately 89% of observed relationships survived to Wave 3. A naive model that predicts survival for everyone therefore achieves high accuracy without learning meaningful patterns.

For this reason, ROC AUC is the main comparison metric in this project.

## Business Interpretation

For an online dating platform such as OKCupid, the findings suggest:

- Online meeting is not enough to explain long-term relationship success.
- Compatibility, relationship quality, commitment, and relationship structure are more informative.
- Product strategy should focus not only on helping users meet, but also on helping users form healthier, more compatible relationships.

A responsible product message would be:

> The right match matters more than where you meet.

Potential product implications:

- improve compatibility-based matching
- include stronger relationship-intent filters
- support long-term goal alignment
- collect optional post-match feedback
- avoid overclaiming that online dating itself creates stronger relationships

## Visual Outputs

Final figures are saved in:

```text
outputs/figures/
```

Important figures include:

- `final_cluster_profile_heatmap.png`
- `final_cluster_sizes.png`
- `final_model_comparison_roc_auc.png`
- `final_top_coefficients.png`
- `final_kmeans_vs_hierarchical_sizes.png`

## How To Run The Project

### 1. Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the notebooks in order

```text
01_data_exploration.ipynb
02_data_cleaning.ipynb
03_clustering.ipynb
04_research_question_model.ipynb
05_final_figures.ipynb
```

The notebooks are designed as a sequential pipeline. Later notebooks rely on processed datasets and result tables generated by earlier notebooks.

The `src/` package can also be imported for reusable workflow components:

```python
from src.cleaning import build_cleaning_outputs
from src.clustering import fit_kmeans_clusters, make_cluster_profile
from src.modeling import make_modeling_dataset, evaluate_model_specs
```

## Requirements

Main libraries:

- pandas
- numpy
- pyreadstat
- matplotlib
- seaborn
- scikit-learn
- scipy
- plotly
- jupyter
- yellowbrick
- adjustText

Note: `notebooks/01_data_exploration.ipynb` imports `missingno`, but it is not currently listed in `requirements.txt`.

## Limitations

This analysis is observational and should not be interpreted causally.

Important limitations:

- Relationship survival is only observed for respondents with known Wave 3 outcomes.
- Wave 3 attrition reduces the available modeling sample.
- The target variable is highly imbalanced.
- Survey responses may contain reporting bias.
- Clustering results depend on feature selection and scaling decisions.
- The primary workflow is still notebook-based, even though reusable helper functions now exist in `src/`.

## Future Improvements

Possible next steps:

- Expand the `src/` modules into a fully scriptable end-to-end pipeline.
- Add command-line entry points for cleaning, clustering, modeling, and plotting.
- Add automated checks for expected dataset shapes and columns.
- Improve handling of survey weights in downstream modeling.
- Explore additional models such as random forests or gradient boosting.
- Use cross-validation for more stable model comparison.
- Add clearer documentation for the construction of the Wave 3 survival target.

## Final Conclusion

This project shows that meeting online has some predictive relationship with long-term survival, but it is not the main story. Relationship quality and relationship structure provide stronger signals.

In practical terms, the findings suggest that successful long-term relationships depend less on the meeting channel itself and more on compatibility, commitment, and relationship dynamics after the match.
