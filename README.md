# 🩺 Diabetes Prediction

Predict whether a patient is likely to have diabetes using machine learning models trained on clinical features.

---

## 📘 Overview

This project builds a machine learning pipeline for diabetes prediction — including data preprocessing, model training, evaluation, and prediction.  
It uses the **Pima Indians Diabetes Dataset** (UCI Repository).

---

## ⚙️ Tech Stack

-   **Language:** Python 3.8+
-   **Libraries:** pandas, numpy, scikit-learn, matplotlib, seaborn, xgboost
-   **Environment:** Jupyter Notebooks or command-line scripts

---

## 🚀 Getting Started

```bash
# Clone repo
git clone https://github.com/kushal-soni01/Diabetes-Predicition.git
cd Diabetes-Predicition

# Setup environment
python -m venv .venv
source .venv/bin/activate     # or .venv\Scripts\activate (Windows)

# Install dependencies
pip install -r requirements.txt
Run training:

bash
Copy code
python src/train.py --data data/diabetes.csv --model random_forest
Make predictions:

bash
Copy code
python src/predict.py --model models/rf_model.joblib --input sample.json
📂 Structure
css
Copy code
data/        - Datasets
notebooks/   - EDA and experiments
src/         - Code (train, predict, utils)
models/      - Trained models
configs/     - Config files
📈 Evaluation
Metrics: Accuracy, Precision, Recall, F1, ROC-AUC.
Example:

bash
Copy code
python src/evaluate.py
🤝 Contributing
Contributions welcome — fork, branch, and open a PR!

🧾 License
MIT License © 2025 Kushal Soni
```
