# 🛡️ Online Fraud Detection (Decision Tree)

This project implements **Decision Tree** algorithms for fraud analysis using the [Online Fraud Dataset](https://www.kaggle.com/datasets).

## 🚀 Features

- **Classification**: Predicts whether a transaction is fraudulent (`isFraud`) based on transaction details.
- **Regression**: Predicts the transaction `amount`.
- **Interactive UI**: Built with Streamlit for easy data exploration and model testing.

## 🛠️ Usage

### **Legacy/Original App**
```bash
python -m streamlit run app.py
```

### **NYC Housing Price Prediction**
```bash
python -m streamlit run nyc_app.py
```

### **Breast Cancer Survival Prediction**
```bash
python -m streamlit run breast_cancer_app.py
```

## 📂 Dataset

The dataset `onlinefraud.csv` is required to run this app. Place it in the root directory.
**Note**: The dataset is excluded from this repository due to its size (>500MB).

## 📦 Requirements

- streamlit
- pandas
- scikit-learn
- matplotlib
- seaborn
