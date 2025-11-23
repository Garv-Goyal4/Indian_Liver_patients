# train_and_save_model.py
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
import joblib

csv_path = "indian_liver_patient.csv"  # adjust path if needed

df = pd.read_csv(csv_path)
# Heuristic target selection
possible_targets = [c for c in df.columns if c.lower() in ['dataset','liver','liver_disease','target','label','status']]
if not possible_targets:
    # fallback: pick a column with values subset {1,2}
    for c in df.columns:
        if set(df[c].dropna().unique()).issubset({1,2}):
            possible_targets.append(c)
if not possible_targets:
    raise ValueError("Couldn't detect target column. Edit the script to specify target column name.")
target_col = possible_targets[0]
print("Using target column:", target_col)

X = df.drop(columns=[target_col])
y = df[target_col].copy()
# Map 1/2 -> 1/0 if applicable (Indian Liver dataset uses 1=patient,2=healthy)
if set(y.dropna().unique()).issubset({1,2}):
    y = y.map({1:1, 2:0})

# Normalize Gender strings if present
if 'Gender' in X.columns:
    X['Gender'] = X['Gender'].astype(str).str.strip().str.lower().map({'male':'Male','female':'Female','m':'Male','f':'Female'}).fillna(X['Gender'])

num_cols = X.select_dtypes(include=['int64','float64']).columns.tolist()
cat_cols = X.select_dtypes(include=['object','category']).columns.tolist()

numeric_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(transformers=[('num', numeric_transformer, num_cols), ('cat', categorical_transformer, cat_cols)])

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42, test_size=0.2)

models_and_params = [
    ("logreg", LogisticRegression(max_iter=1000), {'model__C':[0.1,1,10]}),
    ("rf", RandomForestClassifier(random_state=42), {'model__n_estimators':[50,100], 'model__max_depth':[None,5,10]}),
    ("gb", GradientBoostingClassifier(random_state=42), {'model__n_estimators':[50,100], 'model__learning_rate':[0.05,0.1]}),
    ("svc", SVC(probability=True), {'model__C':[0.1,1,10], 'model__kernel':['rbf','linear']}),
    ("knn", KNeighborsClassifier(), {'model__n_neighbors':[3,5,7]})
]

best_score = -1
best_pipeline = None
best_name = None
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, estimator, params in models_and_params:
    pipe = Pipeline(steps=[('pre', preprocessor), ('model', estimator)])
    print(f"Running GridSearch for {name} ...")
    gs = GridSearchCV(pipe, param_grid=params, scoring='f1', n_jobs=-1, cv=cv, verbose=1)
    gs.fit(X_train, y_train)
    score = gs.best_score_
    print(f"{name} best CV f1: {score:.4f} | best params: {gs.best_params_}")
    if score > best_score:
        best_score = score
        best_pipeline = gs.best_estimator_
        best_name = name

print("\nBest model:", best_name, "with CV f1:", best_score)
y_pred = best_pipeline.predict(X_test)
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
print("Test accuracy:", acc)
print("Test F1:", f1)
print("\nClassification report:\n", classification_report(y_test, y_pred))

model_path = "best_model.joblib"
joblib.dump(best_pipeline, model_path)
print("Saved best pipeline to:", model_path)
