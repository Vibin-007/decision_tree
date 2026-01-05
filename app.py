import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# Set page configuration
st.set_page_config(page_title="Breast Cancer Survival Prediction", layout="wide")

st.title("🎗️ Breast Cancer Survival Prediction (Decision Tree)")
st.markdown("""
This application predicts the **Survival Status** (Alive vs Dead) of breast cancer patients 
based on clinical features using a **Decision Tree Classifier**.
""")

# Load dataset
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('Breast_Cancer.csv')
        # Helper to ensure pyarrow compatibility for Streamlit
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].astype(str)
        return df
    except FileNotFoundError:
        st.error("File 'Breast_Cancer.csv' not found. Please place it in the same directory.")
        return None

df = load_data()

if df is not None:
    # Sidebar navigation
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.selectbox("Choose Mode", ["Data Overview", "Survival Prediction (Classification)"])

    def preprocess_data(df):
        df_p = df.copy()
        le_dict = {}
        
        # Categorical columns to encode
        categorical_cols = [
            'Race', 'Marital Status', 'T Stage ', 'N Stage', '6th Stage', 
            'differentiate', 'Grade', 'A Stage', 'Estrogen Status', 
            'Progesterone Status', 'Status'
        ]
        
        # Note: 'T Stage ' has a trailing space in some versions, treating carefully
        # Normalize column names just in case
        df_p.columns = df_p.columns.str.strip()
        categorical_cols = [c.strip() for c in categorical_cols if c.strip() in df_p.columns]

        for col in categorical_cols:
            le = LabelEncoder()
            df_p[col] = le.fit_transform(df_p[col].astype(str))
            le_dict[col] = le
            
        target = 'Status'
        if target not in df_p.columns:
            st.error(f"Target column '{target}' not found!")
            return None, None, None, None

        X = df_p.drop(columns=[target])
        y = df_p[target]
        
        return X, y, df_p, le_dict

    if app_mode == "Data Overview":
        st.header("📊 Dataset Overview")
        st.write("First 10 rows:")
        st.dataframe(df.head(10))
        
        st.write("Statistics:")
        st.write(df.describe())
        
        st.subheader("Survival Status Distribution")
        if 'Status' in df.columns:
            st.bar_chart(df['Status'].value_counts())
            
        st.subheader("Age Distribution")
        fig, ax = plt.subplots()
        sns.histplot(df['Age'], kde=True, ax=ax)
        st.pyplot(fig)

    elif app_mode == "Survival Prediction (Classification)":
        st.header("🩺 Survival Prediction Model")
        
        X, y, df_p, le_dict = preprocess_data(df)
        
        if X is not None:
            # Check class balance
            st.write(f"Class distribution: {y.value_counts(normalize=True).to_dict()}")
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            col1, col2 = st.columns(2)
            
            with col1:
                max_depth = st.slider("Max Depth", 1, 30, 5)
                min_samples_split = st.slider("Min Samples Split", 2, 50, 20)
                
                if st.button("Train Classifier"):
                    model = DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split, random_state=42)
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    
                    st.success("Model trained successfully!")
                    st.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.4f}")
                    
                    st.text("Classification Report:")
                    st.text(classification_report(y_test, y_pred))
                    
                    st.subheader("Feature Importance")
                    feature_imp = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False).head(10)
                    fig, ax = plt.subplots()
                    sns.barplot(x=feature_imp, y=feature_imp.index, ax=ax, palette="viridis")
                    st.pyplot(fig)

            with col2:
                if 'y_pred' in locals():
                    st.subheader("Confusion Matrix")
                    cm = confusion_matrix(y_test, y_pred)
                    fig_cm, ax_cm = plt.subplots()
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm)
                    ax_cm.set_xlabel('Predicted')
                    ax_cm.set_ylabel('Actual')
                    st.pyplot(fig_cm)
                    
                    st.subheader("Actual vs Predicted (Test Sample)")
                    # Decode target for display
                    if 'Status' in le_dict:
                        actual_labels = le_dict['Status'].inverse_transform(y_test[:10])
                        predicted_labels = le_dict['Status'].inverse_transform(y_pred[:10])
                    else:
                        actual_labels = y_test[:10]
                        predicted_labels = y_pred[:10]
                        
                    results = pd.DataFrame({
                        "Actual Status": actual_labels,
                        "Predicted Status": predicted_labels
                    })
                    st.table(results)
