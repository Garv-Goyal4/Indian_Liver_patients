
# Liver Disease Prediction Model - README

This project provides a Python script that loads a pre-trained machine learning model (saved using `joblib`) and predicts whether a patient is **Healthy** or has **Liver Disease** based on medical input parameters.

---

##  Features

* Accepts patient details through console input
* Loads a pre-trained pipeline (`best_model.joblib`)
* Predicts: **Healthy (0)** or **Liver Disease (1)**
* Displays probability scores for both classes

##  Requirements

Install the following Python libraries:

```
pip install pandas joblib scikit-learn
```

---

##  Project Structure

```
project_folder/
│── best_model.joblib
│── predict.py
│── README.md


## How to Run the Prediction Script

1. Place your trained model file `best_model.joblib` in the same folder as the script.
2. Save the provided script as `predict.py`.
3. Run the script:

```bash
python predict.py
```

4. Enter the patient details when prompted.

---

##  Input Features Required

The model expects the following fields:

* Age
* Gender (Male/Female)
* Total Bilirubin
* Direct Bilirubin
* Alkaline Phosphotase
* Alamine Aminotransferase (ALT)
* Aspartate Aminotransferase (AST)
* Total Proteins
* Albumin
* Albumin and Globulin Ratio

##   Output

==============================
    Prediction Result
==============================
The model predicts: Healthy

Probability Scores:
Healthy (0): 0.8653
Liver Disease (1): 0.1347
==============================

##  Important Notes

* Ensure that `Gender` is provided exactly as expected (Male/Female).
* The model must be trained with the same feature structure.
* If you modify the dataset or pipeline, regenerate `best_model.joblib`.
