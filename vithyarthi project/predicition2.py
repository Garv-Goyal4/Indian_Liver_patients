

import joblib
import pandas as pd

# Load trained model/pipeline
pipe = joblib.load("best_model.joblib")

print("\nEnter Patient Details Below:\n")

age = int(input("Age: "))
gender = input("Gender (Male/Female): ")
total_bilirubin = float(input("Total Bilirubin: "))
direct_bilirubin = float(input("Direct Bilirubin: "))
alk_phos = float(input("Alkaline Phosphotase: "))
alat = float(input("Alamine Aminotransferase (ALT): "))
asat = float(input("Aspartate Aminotransferase (AST): "))
total_protein = float(input("Total Proteins: "))
albumin = float(input("Albumin: "))
ag_ratio = float(input("Albumin and Globulin Ratio: "))

# Create dataframe for prediction
df_new = pd.DataFrame([{
    'Age': age,
    'Gender': gender,
    'Total_Bilirubin': total_bilirubin,
    'Direct_Bilirubin': direct_bilirubin,
    'Alkaline_Phosphotase': alk_phos,
    'Alamine_Aminotransferase': alat,
    'Aspartate_Aminotransferase': asat,
    'Total_Protiens': total_protein,
    'Albumin': albumin,
    'Albumin_and_Globulin_Ratio': ag_ratio
}])

# Predict
pred = pipe.predict(df_new)[0]
prob = pipe.predict_proba(df_new)[0]

print("\n==============================")
print(" Prediction Result ")
print("==============================")

if pred == 1:
    print(" The model predicts: **Liver Disease**")
else:
    print("The model predicts: **Healthy**")


print("\nProbability Scores:")
print(f"Healthy (0): {prob[0]:.4f}")
print(f"Liver Disease (1): {prob[1]:.4f}")
print("==============================\n")
