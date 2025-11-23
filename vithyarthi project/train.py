# train_and_save.py
import os
import sys
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

CSV_PATH = "indian_liver_patient.csv"
MODEL_PATH = "pretrained_liver_model.pkl"
META_PATH = "model_meta.pkl"

# Common target column name variants in liver datasets
POSSIBLE_TARGET_COLS = [
    "Liver_Disease", "Dataset", "dataset", "Liver disease", "LiverDisease", "Target", "target"
]
# Columns to drop if present (IDs / non-features)
POSSIBLE_ID_COLS = ["Patient", "PatientID", "id", "ID", "RowNumber"]

def load_and_prepare(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV file not found at: {path}\nMake sure the path is correct.")

    df = pd.read_csv(path)
    # normalize column names
    df.columns = [c.strip().replace(" ", "_") for c in df.columns]

    print("Columns found in CSV:", df.columns.tolist())

    # find target column
    target_col = None
    for cand in POSSIBLE_TARGET_COLS:
        cand_norm = cand.strip().replace(" ", "_")
        if cand_norm in df.columns:
            target_col = cand_norm
            break

    if target_col is None:
        raise ValueError(
            "Target column not found. Expected one of: "
            f"{POSSIBLE_TARGET_COLS}. Found columns: {df.columns.tolist()}"
        )

    # Drop obvious ID / non-feature columns if present
    for c in POSSIBLE_ID_COLS:
        if c in df.columns:
            print(f"Dropping ID column: {c}")
            df = df.drop(columns=[c])

    # Handle Gender column
    if "Gender" in df.columns:
        le_gender = LabelEncoder()
        df["Gender"] = le_gender.fit_transform(df["Gender"].astype(str))
        # save gender mapping in metadata if needed later
        gender_mapping = dict(zip(le_gender.classes_, le_gender.transform(le_gender.classes_)))
    else:
        # not fatal but warn
        print("Warning: 'Gender' column not found. Proceeding without gender encoding.")
        gender_mapping = None

    # Separate target and features early to avoid altering target with median fill
    y_raw = df[target_col].copy()
    X = df.drop(columns=[target_col])

    # Fill numeric missing with median
    try:
        X = X.fillna(X.median(numeric_only=True))
    except Exception as ex:
        print("Warning during numeric median fill:", ex)

    # For any remaining missing (e.g., non-numeric), attempt forward/back fill
    if X.isnull().any().any():
        X = X.fillna(method="ffill").fillna(method="bfill")

    # Convert any remaining non-numeric feature columns if possible (LabelEncode)
    for col in X.select_dtypes(include=["object", "category"]).columns:
        # If it's already string-like and low-cardinality, encode it
        try:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            print(f"Label-encoded column: {col}")
        except Exception:
            # leave as-is if cannot
            print(f"Could not encode column {col}; leaving as-is.")

    # Prepare target (y) - ensure numeric
    if pd.api.types.is_numeric_dtype(y_raw):
        y = y_raw.astype(int)
    else:
        # label-encode target if it's non-numeric
        le_t = LabelEncoder()
        y = le_t.fit_transform(y_raw.astype(str))
        print("Target values were non-numeric; label-encoded target.")
        target_mapping = dict(zip(le_t.classes_, le_t.transform(le_t.classes_)))
    # if numeric but not integer-like, convert to int safely
    y = pd.Series(y).astype(int)

    # Final sanity checks
    if X.shape[0] != len(y):
        raise ValueError("Number of feature rows does not match number of target rows.")

    return X, y, {"gender_mapping": gender_mapping, "target_mapping": locals().get("target_mapping", None)}

def main():
    try:
        X, y, meta_extra = load_and_prepare(CSV_PATH)
    except Exception as e:
        print("Error when loading/preparing data:", str(e))
        sys.exit(1)

    print("Feature columns used:", X.columns.tolist())
    column_order = X.columns.tolist()

    # 80/20 split (stratify if possible)
    stratify_arg = y if len(set(y)) > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=stratify_arg
    )

    # Train a RandomForest (good baseline)
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate quickly
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    # Save model and metadata
    joblib.dump(model, MODEL_PATH)
    joblib.dump({"columns": column_order, **meta_extra}, META_PATH)
    print(f"Model saved to: {MODEL_PATH}")
    print(f"Metadata saved to: {META_PATH}")

if __name__ == "__main__":
    main()
