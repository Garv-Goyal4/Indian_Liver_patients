# train_and_save.py
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

CSV_PATH = "/mnt/data/indian_liver_patient.csv"
MODEL_PATH = "/mnt/data/pretrained_liver_model.pkl"
# (optional) save the column order so you use same order at inference
META_PATH = "/mnt/data/model_meta.pkl"

def load_and_prepare(path):
    df = pd.read_csv(path)
    # normalize column names
    df.columns = [c.strip().replace(" ", "_") for c in df.columns]

    # Identify target column - adjust if your file uses a different name
    target_col = "Liver_Disease" if "Liver_Disease" in df.columns else "Dataset"
    if target_col not in df.columns:
        raise ValueError(f"Target column not found. Expected 'Liver_Disease' or 'Dataset'. Columns: {df.columns.tolist()}")

    # Encode gender (if exists)
    if "Gender" in df.columns:
        le = LabelEncoder()
        df["Gender"] = le.fit_transform(df["Gender"].astype(str))
    else:
        raise ValueError("Column 'Gender' not found in data.")

    # Fill missing numeric values with median
    df = df.fillna(df.median(numeric_only=True))

    # Split
    X = df.drop(target_col, axis=1)
    y = df[target_col].astype(int)

    return X, y

def main():
    X, y = load_and_prepare(CSV_PATH)

    # keep column order for inference
    column_order = X.columns.tolist()

    # 80/20 split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Train a RandomForest (good baseline)
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate quickly
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    # Save model and metadata
    joblib.dump(model, MODEL_PATH)
    joblib.dump({"columns": column_order}, META_PATH)
    print(f"Model saved to: {MODEL_PATH}")
    print(f"Metadata saved to: {META_PATH}")

if __name__ == "__main__":
    main()
