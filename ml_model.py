import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import accuracy_score, r2_score

# Data preprocessing function
def preprocess_data(df):
    # Add your data cleaning logic, e.g., handling missing values or encoding categorical variables
    df = df.dropna()  # Example: Drop missing values
    return df

# Machine learning model training function
def train_model(X, y, model_type, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    if model_type == "Logistic Regression":
        model = LogisticRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        return model, f"Accuracy: {accuracy:.2f}", y_pred, y_test
    
    elif model_type == "Linear Regression":
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        return model, f"R²: {r2:.2f}", y_pred, y_test
    
    elif model_type == "Random Forest Classifier":
        model = RandomForestClassifier(random_state=random_state)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        return model, f"Accuracy: {accuracy:.2f}", y_pred, y_test
    
    elif model_type == "Random Forest Regressor":
        model = RandomForestRegressor(random_state=random_state)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        return model, f"R²: {r2:.2f}", y_pred, y_test
    
    elif model_type == "Decision Tree Classifier":
        model = DecisionTreeClassifier(random_state=random_state)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        return model, f"Accuracy: {accuracy:.2f}", y_pred, y_test
    
    elif model_type == "Decision Tree Regressor":
        model = DecisionTreeRegressor(random_state=random_state)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        return model, f"R²: {r2:.2f}", y_pred, y_test

# Feature option functions
def plot_histogram(data, column):
    st.write(f"### Histogram of {column}")
    plt.figure(figsize=(10, 5))
    sns.histplot(data[column], kde=True)
    st.pyplot(plt)

def plot_boxplot(data, column):
    st.write(f"### Boxplot of {column}")
    plt.figure(figsize=(10, 5))
    sns.boxplot(data[column])
    st.pyplot(plt)

def plot_scatter(data, col1, col2):
    st.write(f"### Scatter Plot of {col1} vs {col2}")
    plt.figure(figsize=(10, 5))
    sns.scatterplot(x=data[col1], y=data[col2])
    st.pyplot(plt)

def plot_heatmap(data):
    st.write("### Heatmap")
    plt.figure(figsize=(10, 8))
    sns.heatmap(data.corr(), annot=True, cmap='coolwarm', fmt=".2f")
    st.pyplot(plt)

def calculate_vif(data):
    st.write("### VIF Calculation")
    vif_data = pd.DataFrame()
    vif_data["Feature"] = data.columns
    vif_data["VIF"] = [variance_inflation_factor(data.values, i) for i in range(data.shape[1])]
    st.write(vif_data)

def convert_to_categorical(data, column):
    st.write(f"### Converting {column} to Categorical")
    data[column] = data[column].astype('category')
    st.write(data[column].head())

def show_data(data):
    st.write("### Data Preview")
    st.write(data.head())

# Streamlit Application Main
title = st.title("Interactive Machine Learning Interface")

# Upload File
uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])
if uploaded_file:
    data = pd.read_csv(uploaded_file)
    num_rows = st.slider("Select number of rows to display", min_value=1, max_value=len(data), value=5)
    st.write(f"### Data Preview (First {num_rows} rows):", data.head(num_rows))
    
    # Data Cleaning
    data_cleaned = preprocess_data(data)
    num_rows_cleaned = st.slider("Select number of rows to display (after cleaning)", min_value=1, max_value=len(data_cleaned), value=5, key="cleaned_slider")
    st.write(f"### Data Preview After Cleaning (First {num_rows_cleaned} rows):", data_cleaned.head(num_rows_cleaned))

    # Feature Selection
    option = st.selectbox(
        "Select a Option",
        [
            "Plot Histogram",
            "Plot Boxplot",
            "Plot Scatter Plot",
            "Plot Heatmap",
            "Calculate VIF",
            "Convert into Categorical Variable"
        ]
    )
    
    if option == "Plot Histogram":
        column = st.selectbox("Select a Numerical Column", options=data_cleaned.select_dtypes(include=[np.number]).columns)
        if column:
            plot_histogram(data_cleaned, column)

    elif option == "Plot Boxplot":
        column = st.selectbox("Select a Numerical Column", options=data_cleaned.select_dtypes(include=[np.number]).columns)
        if column:
            plot_boxplot(data_cleaned, column)

    elif option == "Plot Scatter Plot":
        col1 = st.selectbox("Select X-axis Numerical Column", options=data_cleaned.select_dtypes(include=[np.number]).columns)
        col2 = st.selectbox("Select Y-axis Numerical Column", options=data_cleaned.select_dtypes(include=[np.number]).columns)
        if col1 and col2:
            plot_scatter(data_cleaned, col1, col2)

    elif option == "Plot Heatmap":
        plot_heatmap(data_cleaned.select_dtypes(include=[np.number]))

    elif option == "Calculate VIF":
        calculate_vif(data_cleaned.select_dtypes(include=[np.number]))

    elif option == "Convert into Categorical Variable":
        column = st.selectbox("Select a Column", options=data_cleaned.columns)
        if column:
            convert_to_categorical(data_cleaned, column)

    # Features and Target Selection
    features = st.multiselect("Select Feature Columns (X)", options=data_cleaned.columns)
    target = st.selectbox("Select Target Column (y)", options=data_cleaned.columns)
    
    if features and target:
        X = data_cleaned[features]
        y = data_cleaned[target]

        X = pd.get_dummies(X, drop_first=True)
        
        # Problem Type Selection
        problem_type = st.radio(
            "Select Problem Type",
            ["Classification", "Regression"]
        )
        
        # Model Selection
        if problem_type == "Classification":
            model_type = st.selectbox(
                "Select Machine Learning Model",
                ["Logistic Regression", "Random Forest Classifier", "Decision Tree Classifier"]
            )
        else:
            model_type = st.selectbox(
                "Select Machine Learning Model",
                ["Linear Regression", "Random Forest Regressor", "Decision Tree Regressor"]
            )
        
        # Train Model
        if st.button("Train Model"):
            model, result, predictions, y_test = train_model(X, y, model_type)
            st.write(f"### {model_type} Training Result: {result}")
            
            # Show Comparison Table
            st.write("### Test Set Predictions vs True Values:")
            comparison_df = pd.DataFrame({
                "True Values": y_test,
                "Predicted Values": predictions
            })
            
            # Display All Results
            st.write(comparison_df)
            
            # Add Download Button
            csv = comparison_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Test Set Predictions",
                data=csv,
                file_name="predictions.csv",
                mime="text/csv"
            )
