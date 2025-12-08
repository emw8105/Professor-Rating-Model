from flask import json
import joblib
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

GRADE_COLS = [
    "A+", "A", "A-", "B+", "B", "B-", "C+", "C", "C-", "D+", "D", "D-", "F", # values contributing to GPA
    "CR", "P", "NC", "NF", "NP", "I", "W" # grade columns that don't contribute to GPA, these will be treated as separate indicators
]

# we'll map the GPA only for actual letter grades
LETTER_GPA = {"A+": 4.0, "A": 4.0, "A-": 3.7, "B+": 3.3, "B": 3.0, "B-": 2.7, "C+": 2.3, "C": 2.0, "C-": 1.7, "D+": 1.3, "D": 1.0, "D-": 0.7, "F": 0.0}

# special cases will be handled separately rather than converted to placeholder GPA values (i.e. NC or W = 0, P = 4.0, etc)
SPECIAL_GRADES = {
    "W": "withdraw",
    "I": "incomplete",
    "CR": "credit",
    "P": "pass",
    "NC": "no_credit",
    "NF": "no_credit",
    "NP": "no_credit"
}

# load the professor data
def load_professor_dataset():
    professor_path = os.path.join("data", "matched_professor_data.json")
    try:
        with open(professor_path, "r", encoding="utf-8") as file:
            print(f"Loading professor dataset {professor_path}")
            prof_dict = json.load(file)
    except Exception as e:
        print(f"Error loading professor dataset {professor_path}: {e}")
        raise

    # instructor_id is the key from the json dataset, i.e. jxc064000
    df = pd.DataFrame.from_dict(prof_dict, orient="index")
    df.index.name = "instructor_id"
    df.reset_index(inplace=True)

    # smooth out any formatting inconsistencies
    df["instructor_id"] = df["instructor_id"].astype(str).str.strip()

    expected_defaults = {
        "quality_rating": np.nan,
        "difficulty_rating": np.nan,
        "would_take_again": np.nan,
        "ratings_count": 0,
        "tags": [],
        "url": None,
    }

    for col, default in expected_defaults.items():
        if col not in df.columns:
            df[col] = default

        # special handling for tags (list field)
        if col == "tags":
            df[col] = df[col].apply(lambda x: x if isinstance(x, list) else default)
        else:
            # everything other var type
            df[col] = df[col].apply(lambda x: x if pd.notna(x) else default)


    # TF-IDF will need space-separated strings so we aggregate tags list into single string and flag if the data didn't have a matched RMP profile
    df["tags_text"] = df["tags"].apply(lambda x: " ".join(x) if isinstance(x, list) else "")
    df["has_rmp"] = df["quality_rating"].notnull()

    return df

# load the grade data, used for extra insights (aggregated values use placeholders for withdrawal for example)
def load_all_grade_data():
    all_grades = []

    # iterate over the grades files in /data
    for filename in os.listdir("data"):
        if filename.startswith("enhanced_grades_") and filename.endswith(".csv"):
            grades_path = os.path.join("data", filename)
            print(f"Loading grades file {grades_path}")
            df = pd.read_csv(grades_path)
            all_grades.append(df)

    if not all_grades:
        raise ValueError("No enhanced_grades_*.csv files found in /data directory.")

    # aggregate all grades into one df
    grades_df = pd.concat(all_grades, ignore_index=True)
    
    # normalize instructor id format so that missing ids are empty strings and no leading/trailing whitespace for easy removal
    grades_df["instructor_id"] = (grades_df["instructor_id"].fillna("").astype(str).str.strip().replace({"nan": "", "None": "", "": ""}))

    # remove records with empty instructor_id, sometimes occurs because student professors or TAs without ids are listed on courses
    before = len(grades_df)
    grades_df = grades_df[grades_df["instructor_id"] != ""] # only keep records with instructor ids
    after = len(grades_df)
    print(f"Grade data before removing empty instructor ids: {before}, after: {after}")

    if before != after:
        print(f"Removed {before - after} records with empty instructor ids")

    return grades_df

# not every grade distribution has the same columns, especially covid-era ones so we need to ensure all expected grade columns exist by filling missing ones with 0s
def check_grade_columns(df):
    for c in GRADE_COLS:
        if c not in df.columns:
            print(f"Column {c} not found, adding with default 0s") # it seems there is some variability in the grade columns between semesters so i used this to detect the ones i was missing
            df[c] = 0
    return df

# we compute mean gpa, percentage of letter grades, percentage of withdrawals, incompletes, dfw rate, and enrollment
def engineer_grade_features(df):
    df = check_grade_columns(df)

    df["enrollment"] = df[GRADE_COLS].sum(axis=1)
    total = df["enrollment"].replace(0, np.nan)  # avoid div-by-zero

    print("Calculating GPA and grade percentages")
    df["gpa_sum"] = sum(df[col] * LETTER_GPA[col] for col in LETTER_GPA)
    df["gpa_count"] = df[list(LETTER_GPA.keys())].sum(axis=1)

    df["mean_gpa"] = df["gpa_sum"] / df["gpa_count"].replace(0, np.nan)

    # we use percentages rather than raw counts to normalize for class size differences
    # we also accumulate +'s and -'s into their base letter grade to more easily compare distributions of different sizes and reduce variability
    df["pct_A"] = (df["A+"] + df["A"] + df["A-"]) / total
    df["pct_B"] = (df["B+"] + df["B"] + df["B-"]) / total
    df["pct_C"] = (df["C+"] + df["C"] + df["C-"]) / total
    df["pct_D"] = (df["D+"] + df["D"] + df["D-"]) / total
    df["pct_F"] = df["F"] / total

    # special grade cases will be used as separate indicators
    df["pct_withdraw"] = df["W"] / total
    df["pct_incomplete"] = df["I"] / total
    df["pct_credit"] = df["CR"] / total
    df["pct_pass"] = df["P"] / total
    df["pct_no_credit"] = (df["NC"] + df["NF"] + df["NP"]) / total

    # dfw rate is a typical metric for assessing difficulty, we'll see how this compares to the RMP difficulty rating for instance
    df["dfw_rate"] = (
        df["D+"] + df["D"] + df["D-"] +
        df["F"] + df["W"]
    ) / total

    df = df.fillna(0)

    return df


# safely computes mode of a pandas series
def compute_series_mode(series, default="unknown"):
    if series is None or len(series) == 0:
        return default
    
    # convert everything to a string, some course ids aren't numeric like 6VXX ids
    s = series.dropna().astype(str)
    if s.empty:
        return default
    
    m = s.mode()
    return m.iloc[0] if not m.empty else default


def compute_weighted_features(g, weighted_feats):
    total_enr = g["enrollment"].sum()
    row = {}

    for f in weighted_feats:
        vals = g[f].fillna(0)
        if total_enr > 0:
            row[f] = np.average(vals, weights=g["enrollment"])
        else:
            row[f] = vals.mean() if not vals.empty else 0.0

    return row, total_enr

def compute_instructor_categorical_stats(g):
    return {
        "instr_sections_taught": len(g),
        "instr_unique_courses": g["Subject"].nunique() if "Subject" in g.columns else 0,
        "instr_subject_mode": compute_series_mode(g["Subject"]) if "Subject" in g.columns else "unknown",
        "instr_level_mode": compute_series_mode(g["Catalog Nbr"]) if "Catalog Nbr" in g.columns else "unknown"
    }

def aggregate_instructor_features(course_df):
    # ensure enrollment exists
    if "enrollment" not in course_df.columns:
        print("Row enrollment not found, computing from grade columns")
        course_df["enrollment"] = course_df[GRADE_COLS].sum(axis=1).fillna(0)

    weighted_feats = ["mean_gpa", "pct_A", "pct_B", "pct_C", "pct_D", "pct_F", "pct_withdraw", "pct_incomplete", "pct_pass", "pct_no_credit", "dfw_rate"]

    grouped = course_df.groupby("instructor_id")
    rows = []

    for instr, g in grouped:
        row = {"instructor_id": instr}
        # print(f"Aggregating features for instructor {instr} with {len(g)} courses")

        # weighted numeric features
        weighted_row, total_enr = compute_weighted_features(g, weighted_feats)
        row.update({f"instr_{k}": v for k, v in weighted_row.items()})
        row["instr_total_students"] = total_enr

        # categorical / count-based stats
        row.update(compute_instructor_categorical_stats(g))

        # variance/consistency features (measure teaching quality stability)
        row["instr_gpa_std"] = g["mean_gpa"].std() if len(g) > 1 else 0.0
        row["instr_gpa_cv"] = (g["mean_gpa"].std() / g["mean_gpa"].mean()) if len(g) > 1 and g["mean_gpa"].mean() > 0 else 0.0
        row["instr_dfw_std"] = g["dfw_rate"].std() if len(g) > 1 else 0.0
        
        # grading leniency (how much do profs give A's vs fail students)
        row["instr_leniency"] = row.get("instr_pct_A", 0) - row.get("instr_pct_F", 0)

        # class size features (large vs small class instructor)
        row["instr_avg_class_size"] = g["enrollment"].mean()
        row["instr_max_class_size"] = g["enrollment"].max()

        # extreme grade percentages to identify large variability in grading
        row["instr_extreme_grades"] = row.get("instr_pct_A", 0) + row.get("instr_pct_F", 0)

        rows.append(row)

    return pd.DataFrame(rows)

def build_feature_sets(final_df):
    """
    Identifies numeric, categorical, and text features to use for modeling.
    Ensures the columns exist.
    """
    numeric = [
        "instr_mean_gpa", "instr_pct_A", "instr_pct_B", "instr_pct_C",
        "instr_pct_D", "instr_pct_F", "instr_pct_withdraw",
        "instr_pct_incomplete", "instr_pct_pass", "instr_pct_no_credit",
        "instr_dfw_rate", "instr_total_students",
        "instr_sections_taught", "instr_unique_courses",
        "instr_gpa_std", "instr_gpa_cv", "instr_dfw_std",
        "instr_leniency", "instr_avg_class_size", "instr_max_class_size",
        "instr_extreme_grades"
    ]
    numeric = [c for c in numeric if c in final_df.columns]

    categorical = ["instr_subject_mode", "instr_level_mode"]
    categorical = [c for c in categorical if c in final_df.columns]

    text_col = "tags_text"
    if text_col not in final_df.columns:
        final_df[text_col] = ""

    return numeric, categorical, text_col


def build_preprocessor(numeric, categorical, text_col):
    """
    Returns the ColumnTransformer used by the model pipeline.
    """
    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
        ("scale", MinMaxScaler())
    ])

    categorical_transformer = Pipeline([
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    text_transformer = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=300))
    ])

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric),
            ("cat", categorical_transformer, categorical),
            ("txt", text_transformer, text_col),
        ],
        remainder="drop"
    )


def train_model(final_df, numeric, categorical, text_col):
    """
    Filters trainable rows and trains a multi-output regression model.
    Returns trained pipeline + test evaluation results.
    """

    # normalize would_take_again from 0-100 to 0-1
    if "would_take_again" in final_df.columns:
        final_df["would_take_again_norm"] = final_df["would_take_again"] / 100.0
    else:
        final_df["would_take_again_norm"] = np.nan

    # target columns
    targets = ["quality_rating", "difficulty_rating", "would_take_again_norm"]
    for t in targets:
        if t not in final_df.columns:
            final_df[t] = np.nan

    # define who is trainable (has RMP labels AND has grade-based features)
    # Check if at least one numeric feature exists (indicates they appear in grades data)
    has_grade_features = final_df[numeric].notnull().any(axis=1) if numeric else pd.Series([False] * len(final_df))
    
    final_df["trainable"] = (
        final_df["has_rmp"].fillna(False) &
        final_df["quality_rating"].notnull() &
        has_grade_features
    )

    train_df = final_df[final_df["trainable"]].dropna(subset=targets)

    print(f"\nTraining dataset:")
    print(f"Total professors: {len(final_df)}")
    print(f"Professors with RMP: {final_df['has_rmp'].sum()}")
    print(f"Professors with grade features: {has_grade_features.sum()}")
    print(f"Trainable (RMP + grades): {final_df['trainable'].sum()}")
    print(f"After dropna on targets: {len(train_df)}")

    if len(train_df) == 0:
        raise ValueError(
            "No trainable data available. Ensure professors have both RMP ratings "
            "and appear in the grade dataset."
        )

    X = train_df[numeric + categorical + [text_col]]
    y = train_df[targets]

    preprocessor = build_preprocessor(numeric, categorical, text_col)

    # we'll use a gradient boosting regressor for multi-output regression to capture professor rating patterns
    # this is ideal for our use case since it can handle non-linear relationships and interactions between features effectively
    model = MultiOutputRegressor(
        GradientBoostingRegressor(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            min_samples_split=20,
            min_samples_leaf=10,
            subsample=0.8,
            random_state=42
        )
    )

    pipeline = Pipeline([
        ("preprocess", preprocessor),
        ("model", model)
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("\nTraining model...")
    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)

    # evaluation metrics packaged in a dict
    eval_results = {}
    for i, col in enumerate(y.columns):
        eval_results[col] = {
            "MAE": mean_absolute_error(y_test.iloc[:, i], preds[:, i]),
            "RMSE": np.sqrt(mean_squared_error(y_test.iloc[:, i], preds[:, i])),
            "R2": r2_score(y_test.iloc[:, i], preds[:, i])
        }

    return pipeline, eval_results


def generate_predictions(final_df, pipeline, numeric, categorical, text_col):
    """
    Produces predictions for all instructors using the trained model.
    """
    X_all = final_df[numeric + categorical + [text_col]].fillna(0)
    preds = pipeline.predict(X_all)

    final_df["pred_quality"] = preds[:, 0]
    final_df["pred_difficulty"] = preds[:, 1]
    final_df["pred_would_take_again"] = preds[:, 2]

    return final_df


def analyze_feature_importance(pipeline, numeric, categorical, text_col, top_n=15):
    """
    Extract and display feature importance from the trained model.
    Helps identify which grade features actually matter.
    """
    print("\n" + "="*60)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("="*60)
    
    # Get feature names after preprocessing
    preprocessor = pipeline.named_steps["preprocess"]
    feature_names = []
    
    # numeric features
    feature_names.extend(numeric)

    # categorical features (one-hot encoded)
    if categorical:
        cat_encoder = preprocessor.named_transformers_["cat"].named_steps["onehot"]
        cat_features = cat_encoder.get_feature_names_out(categorical)
        feature_names.extend(cat_features)

    # text features (TF-IDF)
    if text_col:
        tfidf = preprocessor.named_transformers_["txt"].named_steps["tfidf"]
        tfidf_features = [f"tag_{word}" for word in tfidf.get_feature_names_out()]
        feature_names.extend(tfidf_features)
    
    model = pipeline.named_steps["model"]
    targets = ["quality_rating", "difficulty_rating", "would_take_again_norm"]
    
    for idx, target in enumerate(targets):
        estimator = model.estimators_[idx]
        importances = estimator.feature_importances_
        
        feat_imp_df = pd.DataFrame({
            "feature": feature_names,
            "importance": importances
        }).sort_values("importance", ascending=False).head(top_n)
        
        print(f"\n--- Top {top_n} Features for {target} ---")
        for i, row in feat_imp_df.iterrows():
            print(f"  {row['feature']:40s} {row['importance']:.4f}")
    
    print("\n" + "="*60)

def main():
    prof_df = load_professor_dataset()
    grades_df = load_all_grade_data()

    # complete record of grade dist per course per semester, RMP ratings, agg grade ratings, tags, difficulty, would take again %, etc., merge grade and professor data
    merged_df = grades_df.merge(prof_df, on="instructor_id", how="left")

    # compute course-level grade features
    merged_df = engineer_grade_features(merged_df)

    # aggregate instructor features and then merge back with professor metadata
    df = aggregate_instructor_features(merged_df)
    final_df = prof_df.merge(df, on="instructor_id", how="left")

    numeric, categorical, text_col = build_feature_sets(final_df)

    pipeline, eval_results = train_model(final_df, numeric, categorical, text_col)

    print("\nModel Evaluation:")
    for target, metrics in eval_results.items():
        print(f"\n--- {target} ---")
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")

    analyze_feature_importance(pipeline, numeric, categorical, text_col, top_n=15)

    final_df = generate_predictions(final_df, pipeline, numeric, categorical, text_col)
    
    final_df.to_csv("results_dataframe.csv", index=False)
    print("\nSaved results_dataframe.csv")
    joblib.dump(pipeline, "model_pipeline.pkl")
    print("Saved model_pipeline.pkl")


if __name__ == "__main__":
    main()