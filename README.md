# 🩺 Breast Cancer Survival Predictor (Decision Tree)

A Streamlit application that predicts the survival status of breast cancer patients using a Decision Tree model. This tool allows medical professionals to analyze patient data and estimate survival outcomes.

## 📊 Features

- **Survival Prediction**: Classification of patients into "Alive" or "Dead" categories.
- **Interactive Dashboard**: Real-time model training and prediction based on inputs like Tumor Size, Stage, and Grade.
- **Visualizations**: Confusion Matrix, Feature Importance charts, and Decision Tree structure.
- **User-Friendly Interface**: Clean and intuitive UI built with Streamlit.

## 🛠️ Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Vibin-007/decision_tree.git
   cd decision_tree
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:
   ```bash
   streamlit run app.py
   ```

## 📁 Project Structure

- `app.py`: Main application file containing the Streamlit interface and logic.
- `Breast_Cancer.csv`: Dataset containing patient attributes like Age, Race, Marital Status, and Tumor Stage.
- `decision_tree_analysis.ipynb`: Jupyter notebook for exploratory data analysis and model experimentation.
- `requirements.txt`: List of Python dependencies.

## 📈 Model Information

The model uses **Decision Tree Classification** to predict survival based on:
- **T Stage, N Stage, 6th Stage**
- **Tumor Size**
- **Grade & Differentiation**
- **Age, Race, Marital Status**
- **Estrogen & Progesterone Status**
