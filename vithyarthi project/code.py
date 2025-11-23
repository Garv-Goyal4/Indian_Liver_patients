import pandas as pd

# Load the dataset (put your CSV file in the same folder)
df = pd.read_csv("indian_liver_patient.csv")

print("First 5 rows:")
print(df.head())

print("\nDataset Shape:", df.shape)


import matplotlib.pyplot as plt

# Age distribution
plt.hist(df['Age'])
plt.xlabel("Age")
plt.ylabel("Count")
plt.title("Age Distribution of Liver Patients")
plt.show()

# Gender distribution
df['Gender'].value_counts().plot(kind='bar')
plt.title("Gender Count")
plt.xlabel("Gender")
plt.ylabel("Count")
plt.show()

