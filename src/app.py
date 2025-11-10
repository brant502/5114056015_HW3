import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
from PIL import Image

# 設置頁面配置
st.set_page_config(
    page_title="垃圾郵件分類系統",
    layout="wide"
)

# 標題
st.title("垃圾郵件分類系統 - Phase 1")
st.markdown("使用 SVM 進行垃圾郵件分類的機器學習模型")

@st.cache_resource
def load_model():
    """載入預訓練的模型和向量器"""
    try:
        model = joblib.load('models/spam_model.joblib')
        vectorizer = joblib.load('models/vectorizer.joblib')
        return model, vectorizer
    except:
        st.error('無法載入模型。請確保已執行 train_model.py 生成模型文件。')
        return None, None

@st.cache_data
def load_evaluation_data():
    """載入預先計算的評估數據"""
    try:
        # 載入標籤分布數據
        with open('models/label_distribution.json', 'r') as f:
            label_distribution = json.load(f)
        
        # 載入混淆矩陣圖
        confusion_matrix_img = Image.open('models/confusion_matrix.png')

        # 載入分類報告（metrics）
        try:
            with open('models/metrics.json', 'r', encoding='utf-8') as mf:
                metrics = json.load(mf)
        except Exception:
            metrics = None
        
        return label_distribution, confusion_matrix_img, metrics
    except:
        st.error('無法載入評估數據。請確保已執行 train_model.py 生成必要文件。')
        return None, None, None

def main():
    # 載入模型
    model, vectorizer = load_model()
    if model is None or vectorizer is None:
        return
    
    # 載入評估數據
    label_distribution, confusion_matrix_img, metrics = load_evaluation_data()
    if label_distribution is None or confusion_matrix_img is None:
        return
        
    # 顯示數據統計
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("數據集概況")
        total_samples = sum(label_distribution.values())
        st.write(f"總數據量：{total_samples} 條")
        st.write(f"垃圾郵件：{label_distribution['spam']} 條")
        st.write(f"正常郵件：{label_distribution['ham']} 條")
        
    with col2:
        st.subheader("標籤分布")
        df_distribution = pd.DataFrame.from_dict(label_distribution, orient='index')
        st.bar_chart(df_distribution)
    
    # 顯示模型評估結果
    st.subheader("模型評估")
    col3, col4 = st.columns(2)
    
    with col3:
        st.image(confusion_matrix_img, caption='混淆矩陣', use_column_width=True)
    
    with col4:
        st.subheader("分類指標 (precision / recall / f1)")
        if metrics is not None:
            # 轉換 metrics 為 DataFrame（只取 spam 和 ham 的 precision/recall/f1）
            try:
                # metrics example keys: '0', '1', 'accuracy', 'macro avg', 'weighted avg'
                rows = {}
                # prefer explicit keys 'ham'/'spam' if present, otherwise map '0'->ham '1'->spam
                if 'ham' in metrics and 'spam' in metrics:
                    keys = ['ham', 'spam']
                else:
                    # map numeric keys
                    keys = []
                    if '0' in metrics:
                        keys.append('0')
                    if '1' in metrics:
                        keys.append('1')

                for k in keys:
                    m = metrics.get(k, {})
                    rows[k] = {
                        'precision': m.get('precision', 0),
                        'recall': m.get('recall', 0),
                        'f1-score': m.get('f1-score', 0)
                    }
                df_metrics = pd.DataFrame.from_dict(rows, orient='index')
                st.dataframe(df_metrics)
                # 畫條形圖
                st.bar_chart(df_metrics)
            except Exception as e:
                st.error(f'無法解析 metrics.json：{e}')
        else:
            st.info('metrics.json 尚未生成，請先執行模型訓練以生成預計算資料。')
    
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
        
        # 使用更醒目的方式顯示結果
        col5, col6 = st.columns(2)
        with col5:
            st.markdown(f"### 預測結果：{result}")
        with col6:
            st.markdown(f"### 置信度：{confidence:.2%}")
            
        # 增加結果解釋
        if prediction[0] == 1:
            st.warning('⚠️ 此郵件被識別為垃圾郵件，建議謹慎處理！')
        else:
            st.success('✅ 此郵件看起來是正常的郵件。')

if __name__ == '__main__':
    main()