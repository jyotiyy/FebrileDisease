<h1 align="center">Febrile Disease Predictor</h1>

<p align="center">
  Bayesian Network Based Disease Prediction System
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.x-blue?style=for-the-badge">
  <img src="https://img.shields.io/badge/Bayesian-Network-green?style=for-the-badge">
  <img src="https://img.shields.io/badge/Status-Completed-success?style=for-the-badge">
</p>

---

## Overview

This project predicts possible **febrile diseases** based on patient symptoms using a **Bayesian Network**. The model estimates disease probabilities from the given evidence and returns the most likely diagnosis.

---

## Features

- Bayesian Network based inference
- Symptom-based disease prediction
- Probability estimation
- Fast and lightweight implementation
- Easy to extend with additional diseases

---

## Workflow

```mermaid
flowchart LR

A[Patient Symptoms]
--> B[Input Processing]

B --> C[Bayesian Network]

C --> D[Probability Calculation]

D --> E[Most Likely Disease]

E --> F[Prediction Result]
```

---

<details>
<summary><b>Prediction Pipeline</b></summary>

```text
Patient
   │
   ▼
Enter Symptoms
   │
   ▼
Bayesian Inference
   │
   ▼
Probability Computation
   │
   ▼
Disease Prediction
```

</details>

---

## Technologies Used

- Python
- Bayesian Network
- Probability Theory

---

## Output

The system:

- Accepts symptoms as input
- Computes posterior probabilities
- Predicts the most probable febrile disease

---

## Future Improvements

- Larger medical dataset
- Graphical User Interface
- Improved Bayesian model
- Confidence score visualization

---

<p align="center">
Developed as an educational project demonstrating probabilistic disease prediction using Bayesian Networks.
</p>
