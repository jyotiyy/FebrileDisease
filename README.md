<h1 align="center">Febrile Disease Predictor</h1>

<p align="center">
Bayesian Network Based Disease Prediction with Hill Climbing Treatment Optimization
</p>

<p align="center">
<img src="https://img.shields.io/badge/Python-3.x-blue?style=for-the-badge">
<img src="https://img.shields.io/badge/Bayesian-Network-green?style=for-the-badge">
<img src="https://img.shields.io/badge/Hill%20Climbing-Optimization-orange?style=for-the-badge">
<img src="https://img.shields.io/badge/Status-Completed-success?style=for-the-badge">
</p>

---

## Overview

This project predicts **febrile diseases** from patient symptoms using a **Bayesian Network** and recommends the **most suitable treatment** using the **Hill Climbing Search Algorithm**.

---

## Features

- Bayesian Network based disease prediction
- Symptom probability estimation
- Hill Climbing treatment optimization
- Fast and lightweight implementation

---

## Workflow

```mermaid
flowchart TD

A([Patient])

A --> B[Enter Symptoms]

B --> C[Data Preprocessing]

C --> D[Bayesian Network]

D --> E[Disease Probability Estimation]

E --> F[Predicted Disease]

F --> G[Available Treatments]

G --> H[Hill Climbing Search]

H --> I[Optimal Treatment]

I --> J([Prediction Report])
```

---

## Project Pipeline

```text
Patient
   │
   ▼
Symptoms
   │
   ▼
Bayesian Network
   │
   ▼
Disease Prediction
   │
   ▼
Treatment Options
   │
   ▼
Hill Climbing Search
   │
   ▼
Optimal Treatment
```

---

## Technologies Used

- Python
- Bayesian Network
- Hill Climbing Search

---

## Output

- Predicted disease
- Disease probability
- Optimized treatment recommendation

---

<p align="center">
Developed as an educational project demonstrating probabilistic disease prediction and AI-based treatment optimization.
</p>
