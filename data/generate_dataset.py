
import pandas as pd
import numpy as np

np.random.seed(42)

def generate_disease_samples(disease, n):
    
    records = []

    for _ in range(n):
        age = np.random.randint(5, 75)

        if disease == "Typhoid":
          
            fever       = 1
            chills      = np.random.choice([0, 1], p=[0.6, 0.4])
            headache    = np.random.choice([0, 1], p=[0.2, 0.8])
            nausea      = np.random.choice([0, 1], p=[0.3, 0.7])
            body_pain   = np.random.choice([0, 1], p=[0.4, 0.6])
            fatigue     = np.random.choice([0, 1], p=[0.2, 0.8])
            rash        = np.random.choice([0, 1], p=[0.85, 0.15]) 
            vomiting    = np.random.choice([0, 1], p=[0.4, 0.6])
            platelets   = np.random.randint(100000, 200000)

        elif disease == "Malaria":
           
            fever       = 1
            chills      = np.random.choice([0, 1], p=[0.1, 0.9])
            headache    = np.random.choice([0, 1], p=[0.2, 0.8])
            nausea      = np.random.choice([0, 1], p=[0.3, 0.7])
            body_pain   = np.random.choice([0, 1], p=[0.3, 0.7])
            fatigue     = np.random.choice([0, 1], p=[0.2, 0.8])
            rash        = np.random.choice([0, 1], p=[0.95, 0.05])
            vomiting    = np.random.choice([0, 1], p=[0.4, 0.6])
            platelets   = np.random.randint(40000, 120000)           

        elif disease == "Viral_Fever":
            fever       = np.random.choice([0, 1], p=[0.1, 0.9])
            chills      = np.random.choice([0, 1], p=[0.5, 0.5])
            headache    = np.random.choice([0, 1], p=[0.3, 0.7])
            nausea      = np.random.choice([0, 1], p=[0.5, 0.5])
            body_pain   = np.random.choice([0, 1], p=[0.3, 0.7])
            fatigue     = np.random.choice([0, 1], p=[0.3, 0.7])
            rash        = np.random.choice([0, 1], p=[0.7, 0.3])
            vomiting    = np.random.choice([0, 1], p=[0.6, 0.4])
            platelets   = np.random.randint(150000, 300000)

        elif disease == "Rickettsial":
            
            fever       = 1
            chills      = np.random.choice([0, 1], p=[0.3, 0.7])
            headache    = np.random.choice([0, 1], p=[0.1, 0.9])
            nausea      = np.random.choice([0, 1], p=[0.5, 0.5])
            body_pain   = np.random.choice([0, 1], p=[0.2, 0.8])
            fatigue     = np.random.choice([0, 1], p=[0.2, 0.8])
            rash        = np.random.choice([0, 1], p=[0.1, 0.9])   
            vomiting    = np.random.choice([0, 1], p=[0.6, 0.4])
            platelets   = np.random.randint(60000, 150000)

        records.append({
            "age": age, "fever": fever, "chills": chills,
            "headache": headache, "nausea": nausea, "body_pain": body_pain,
            "fatigue": fatigue, "rash": rash, "vomiting": vomiting,
            "platelets": platelets, "disease": disease
        })

    return records

samples = []
for disease, n in [("Typhoid", 300), ("Malaria", 300),
                   ("Viral_Fever", 300), ("Rickettsial", 300)]:
    samples.extend(generate_disease_samples(disease, n))

df = pd.DataFrame(samples).sample(frac=1, random_state=42).reset_index(drop=True)
df.to_csv("/home/claude/febrile_dss/data/dataset.csv", index=False)
print(f"Dataset generated: {len(df)} rows")
print(df["disease"].value_counts())
