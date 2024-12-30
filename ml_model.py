import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
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
        return model, f"Accuracy: {accuracy:.2f}"
    
    elif model_type == "Linear Regression":
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        return model, f"R²: {r2:.2f}"
    
    elif model_type == "Random Forest Classifier":
        model = RandomForestClassifier(random_state=random_state)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        return model, f"Accuracy: {accuracy:.2f}"
    
    elif model_type == "Random Forest Regressor":
        model = RandomForestRegressor(random_state=random_state)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        return model, f"R²: {r2:.2f}"

# Streamlit 應用主體
st.title("機器學習模型互動界面")

# 上傳檔案
uploaded_file = st.file_uploader("請上傳資料檔案 (CSV)", type=["csv"])
if uploaded_file:
    data = pd.read_csv(uploaded_file)
    num_rows = st.slider("選擇要顯示的筆數", min_value=1, max_value=len(data), value=5)
    st.write(f"### 資料預覽（顯示前 {num_rows} 筆）：", data.head(num_rows))
    
    # 資料前處理
    data = preprocess_data(data)
    num_rows = st.slider("選擇要顯示的筆數", min_value=1, max_value=len(data), value=5)
    st.write(f"### 清理後的資料（顯示前 {num_rows} 筆）：", data.head(num_rows))
    st.write("### 清理後的資料：", data.head())
    
    # 特徵與標籤選擇
    features = st.multiselect("選擇特徵欄位 (X)", options=data.columns)
    target = st.selectbox("選擇目標欄位 (y)", options=data.columns)
    
    if features and target:
        X = data[features]
        y = data[target]
        
        # 模型選擇
        model_type = st.selectbox(
            "選擇機器學習模型",
            ["Logistic Regression", "Linear Regression", "Random Forest Classifier", "Random Forest Regressor"]
        )
        
        # 訓練模型
        if st.button("開始訓練"):
            model, result = train_model(X, y, model_type)
            st.write(f"### {model_type} 訓練結果：{result}")
