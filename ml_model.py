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

# 資料前處理函數
def preprocess_data(df):
    # 添加你的資料清理邏輯，例如處理缺失值或編碼分類變數
    df = df.dropna()  # 示例：刪除缺失值
    return df

# 機器學習模型訓練函數
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

# 功能選項函數
def plot_histogram(data, column):
    st.write(f"### {column} 的直方圖")
    plt.figure(figsize=(10, 5))
    sns.histplot(data[column], kde=True)
    st.pyplot(plt)

def plot_boxplot(data, column):
    st.write(f"### {column} 的盒狀圖")
    plt.figure(figsize=(10, 5))
    sns.boxplot(data[column])
    st.pyplot(plt)

def plot_scatter(data, col1, col2):
    st.write(f"### {col1} vs {col2} 的散點圖")
    plt.figure(figsize=(10, 5))
    sns.scatterplot(x=data[col1], y=data[col2])
    st.pyplot(plt)

def plot_heatmap(data):
    st.write("### 熱力圖")
    plt.figure(figsize=(10, 8))
    sns.heatmap(data.corr(), annot=True, cmap='coolwarm', fmt=".2f")
    st.pyplot(plt)

def calculate_vif(data):
    st.write("### VIF 指標計算")
    vif_data = pd.DataFrame()
    vif_data["Feature"] = data.columns
    vif_data["VIF"] = [variance_inflation_factor(data.values, i) for i in range(data.shape[1])]
    st.write(vif_data)

def convert_to_categorical(data, column):
    st.write(f"### 將 {column} 轉換為類別型變數")
    data[column] = data[column].astype('category')
    st.write(data[column].head())

def show_data(data):
    st.write("### 資料預覽")
    st.write(data.head())

# Streamlit 應用主體
st.title("機器學習模型互動界面")

# 上傳檔案
uploaded_file = st.file_uploader("請上傳資料檔案 (CSV)", type=["csv"])
if uploaded_file:
    data = pd.read_csv(uploaded_file)
    num_rows = st.slider("選擇要顯示的筆數", min_value=1, max_value=len(data), value=5)
    st.write(f"### 資料預覽（顯示前 {num_rows} 筆）：", data.head(num_rows))
    
    # 資料清理
    data_cleaned = preprocess_data(data)
    num_rows_cleaned = st.slider("選擇要顯示的清理後資料筆數", min_value=1, max_value=len(data_cleaned), value=5, key="cleaned_slider")
    st.write(f"### 清理後的資料預覽（顯示前 {num_rows_cleaned} 筆）：", data_cleaned.head(num_rows_cleaned))


    # 功能選項
    option = st.selectbox(
        "選擇一個功能",
        [
            "Show data",
            "Plot Histogram",
            "Plot Boxplot",
            "Plot Scatter Plot",
            "Plot Heatmap",
            "Calculate VIF",
            "Convert into Categorical Variable"
        ]
    )

    if option == "Show data":
        show_data(data_cleaned)

    elif option == "Plot Histogram":
        column = st.selectbox("選擇一個數值欄位", options=data_cleaned.select_dtypes(include=[np.number]).columns)
        if column:
            plot_histogram(data_cleaned, column)

    elif option == "Plot Boxplot":
        column = st.selectbox("選擇一個數值欄位", options=data_cleaned.select_dtypes(include=[np.number]).columns)
        if column:
            plot_boxplot(data_cleaned, column)

    elif option == "Plot Scatter Plot":
        col1 = st.selectbox("選擇 X 軸數值欄位", options=data_cleaned.select_dtypes(include=[np.number]).columns)
        col2 = st.selectbox("選擇 Y 軸數值欄位", options=data_cleaned.select_dtypes(include=[np.number]).columns)
        if col1 and col2:
            plot_scatter(data_cleaned, col1, col2)

    elif option == "Plot Heatmap":
        plot_heatmap(data_cleaned.select_dtypes(include=[np.number]))

    elif option == "Calculate VIF":
        calculate_vif(data_cleaned.select_dtypes(include=[np.number]))

    elif option == "Convert into Categorical Variable":
        column = st.selectbox("選擇一個欄位", options=data_cleaned.columns)
        if column:
            convert_to_categorical(data_cleaned, column)

    
    # 特徵與標籤選擇
    features = st.multiselect("選擇特徵欄位 (X)", options=data_cleaned.columns)
    target = st.selectbox("選擇目標欄位 (y)", options=data_cleaned.columns)
    
    if features and target:
        X = data_cleaned[features]
        y = data_cleaned[target]

        X = pd.get_dummies(X, drop_first=True)
        
        # 問題類型選擇
        problem_type = st.radio(
            "選擇問題類型",
            ["分類問題", "回歸問題"]
        )
        
        # 根據問題類型顯示模型選項
        if problem_type == "分類問題":
            model_type = st.selectbox(
                "選擇機器學習模型",
                ["Logistic Regression", "Random Forest Classifier", "Decision Tree Classifier"]
            )
        else:
            model_type = st.selectbox(
                "選擇機器學習模型",
                ["Linear Regression", "Random Forest Regressor", "Decision Tree Regressor"]
            )
        
        # 訓練模型
        if st.button("開始訓練"):
            model, result, predictions, y_test = train_model(X, y, model_type)
            st.write(f"### {model_type} 訓練結果：{result}")
            # 訓練模型並保存結果到 session_state
            model, result, predictions, y_test = train_model(X, y, model_type)
            st.session_state["model_result"] = result
            st.session_state["predictions"] = predictions
            st.session_state["y_test"] = y_test
        
            # 顯示對比表格
            st.write("### 測試集預測 vs 真實值：")
            comparison_df = pd.DataFrame({
                "真實值": st.session_state["y_test"],
                "預測值": st.session_state["predictions"]
            })
        
            # 顯示所有結果
            st.write(comparison_df)
        
            # 添加下載功能
            csv = comparison_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="下載測試集預測結果",
                data=csv,
                file_name="predictions.csv",
                mime="text/csv"
            )
