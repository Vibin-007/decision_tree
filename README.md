# 🩺 Breast Cancer Survival Prediction (Decision Tree)

This project implements a **Decision Tree Classifier** to predict the survival status of breast cancer patients based on medical attributes.

## 🚀 Features

- **Survival Prediction**: Classifies patients as 'Alive' or 'Dead' based on tumour size, stage, and other factors.
- **Interactive Dashboard**:
    - **Data Explorer**: View dataset statistics and raw values.
    - **Model Training**: Train the model in real-time.
    - **Visualizations**: Confusion Matrix, Feature Importance, and Decision Tree diagram.

## 🛠️ Usage

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

<<<<<<< HEAD
2. **Run the App**:
   ```bash
   python -m streamlit run app.py
   ```

## 📁 Project Structure

- `app.py`: Streamlit application file.
- `decision_tree_analysis.ipynb`: Jupyter notebook for in-depth analysis.
- `Breast_Cancer.csv`: Dataset used for training.
- `requirements.txt`: Python package dependencies.
=======
### **Breast Cancer Survival Prediction**
```bash
python -m streamlit run breast_cancer_app.py
```
>>>>>>> b3e36d40947da6fc56d6694433f473f3a7b8a3e3

## 📂 Dataset

The project uses `Breast_Cancer.csv`. It contains patient details like Age, Race, Marital Status, T Stage, N Stage, etc.

## 📦 Requirements

- streamlit
- pandas
- scikit-learn
- matplotlib
- seaborn
