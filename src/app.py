import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import time

# 設置頁面配置
st.set_page_config(
    page_title="垃圾郵件分類系統",
    layout="wide"
)

# 標題
st.title("垃圾郵件分類系統 - Phase 1")
st.markdown("使用 SVM 進行垃圾郵件分類的機器學習模型")

@st.cache_data
def load_data():
    """載入數據"""
    url = 'https://raw.githubusercontent.com/PacktPublishing/Hands-On-Artificial-Intelligence-for-Cybersecurity/refs/heads/master/Chapter03/datasets/sms_spam_no_header.csv'
    df = pd.read_csv(url, names=['label', 'text'])
    return df

def prepare_features(df):
    """特徵工程"""
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(df['text'])
    y = (df['label'] == 'spam').astype(int)
    return X, y, vectorizer

def train_model(X_train, y_train):
    """訓練SVM模型"""
    svm = SVC(kernel='rbf', probability=True)
    with st.spinner('模型訓練中...'):
        svm.fit(X_train, y_train)
    return svm

def plot_confusion_matrix(y_test, y_pred):
    """繪製混淆矩陣"""
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    plt.title('混淆矩陣')
    plt.ylabel('真實標籤')
    plt.xlabel('預測標籤')
    return fig

def plot_label_distribution(df):
    """繪製標籤分布"""
    fig, ax = plt.subplots(figsize=(8, 6))
    df['label'].value_counts().plot(kind='bar')
    plt.title('郵件類型分布')
    plt.ylabel('數量')
    plt.xlabel('類型')
    return fig

def main():
    # 載入數據
    try:
        df = load_data()
        st.success('數據載入成功！')
        
        # 顯示數據統計
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("數據集概況")
            st.write(f"總數據量：{df.shape[0]} 條")
            st.write(f"特徵數量：{df.shape[1]} 個")
            
        with col2:
            st.subheader("標籤分布")
            st.write(pd.DataFrame(df['label'].value_counts()).T)
        
        # 顯示標籤分布圖
        st.subheader("數據可視化")
        col3, col4 = st.columns(2)
        
        with col3:
            fig1 = plot_label_distribution(df)
            st.pyplot(fig1)
        
        # 模型訓練和評估
        if st.button('開始訓練模型'):
            # 特徵工程
            X, y, vectorizer = prepare_features(df)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # 訓練模型
            model = train_model(X_train, y_train)
            st.success('模型訓練完成！')
            
            # 模型評估
            y_pred = model.predict(X_test)
            
            # 顯示評估報告
            st.subheader("模型評估結果")
            report = classification_report(y_test, y_pred, output_dict=True)
            df_report = pd.DataFrame(report).transpose()
            st.dataframe(df_report)
            
            # 顯示混淆矩陣
            with col4:
                fig2 = plot_confusion_matrix(y_test, y_pred)
                st.pyplot(fig2)
            
            # 互動式預測
            st.subheader("郵件分類測試")
            test_text = st.text_area("輸入要測試的郵件內容：",
                                   "Enter your message here...")
            
            if st.button('進行預測'):
                # 文本預處理和預測
                X_test = vectorizer.transform([test_text])
                prediction = model.predict(X_test)
                probability = model.predict_proba(X_test)
                
                # 顯示預測結果
                result = "垃圾郵件 ⚠️" if prediction[0] == 1 else "正常郵件 ✅"
                confidence = probability[0][1] if prediction[0] == 1 else probability[0][0]
                
                st.write(f"預測結果：{result}")
                st.write(f"置信度：{confidence:.2%}")
                
    except Exception as e:
        st.error(f'發生錯誤：{str(e)}')

if __name__ == '__main__':
    main()