import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

st.title("🤖 Fully Automated AI Data Analyst")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    st.subheader("📊 Dataset Preview")
    st.write(df.head())
    
    st.subheader("📦 Dataset Shape")
    st.write(df.shape)
    
    st.subheader("🧹 Missing Values")
    st.write(df.isnull().sum())
    
    st.subheader("📊 Statistical Summary")
    st.write(df.describe())
    
    # Correlation Heatmap
    st.subheader("🔎 Correlation Heatmap")
    fig, ax = plt.subplots()
    ax.imshow(df.corr(numeric_only=True))
    st.pyplot(fig)
    
    # Machine Learning Section
    numeric_cols = df.select_dtypes(include=['number']).columns
    
    if len(numeric_cols) >= 2:
        target = st.selectbox("Select Target Column", numeric_cols)
        features = st.multiselect("Select Feature Columns", numeric_cols, default=numeric_cols[:-1])
        
        if st.button("Run ML Model"):
            X = df[features]
            y = df[target]
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            models = {
                "Linear Regression": LinearRegression(),
                "Random Forest": RandomForestRegressor(random_state=42)
            }
            
            best_score = -1
            best_model = None
            
            for name, model in models.items():
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                score = r2_score(y_test, preds)
                st.write(f"{name} R2 Score: {score:.4f}")
                
                if score > best_score:
                    best_score = score
                    best_model = name
            
            st.success(f"🏆 Best Model: {best_model} (R2: {best_score:.4f})")