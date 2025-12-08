from flask import json
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
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


if __name__ == "__main__":
    prof_df = load_professor_dataset()
    grades_df = load_all_grade_data()

    # complete record of grade dist per course per semester, RMP ratings, agg grade ratings, tags, difficulty, would take again %, etc., merge grade and professor data
    merged_df = grades_df.merge(prof_df, on="instructor_id", how="left")

    # compute course-level grade features
    merged_df = engineer_grade_features(merged_df)