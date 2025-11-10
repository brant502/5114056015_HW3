import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import json

def train_and_save_model():
    # 載入數據
    print("載入數據...")
    url = 'https://raw.githubusercontent.com/PacktPublishing/Hands-On-Artificial-Intelligence-for-Cybersecurity/refs/heads/master/Chapter03/datasets/sms_spam_no_header.csv'
    df = pd.read_csv(url, names=['label', 'text'])
    
    # 特徵工程
    print("處理特徵...")
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(df['text'])
    y = (df['label'] == 'spam').astype(int)
    
    # 分割數據
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 訓練模型
    print("訓練模型...")
    model = SVC(kernel='rbf', probability=True)
    model.fit(X_train, y_train)
    
    # 評估模型
    print("評估模型...")
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    # 創建 models 目錄（如果不存在）
    if not os.path.exists('models'):
        os.makedirs('models')
    
    # 保存模型和向量器
    print("保存模型...")
    joblib.dump(model, 'models/spam_model.joblib')
    joblib.dump(vectorizer, 'models/vectorizer.joblib')
    
    # 生成並保存評估圖表
    print("生成評估圖表...")
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('混淆矩陣')
    plt.ylabel('真實標籤')
    plt.xlabel('預測標籤')
    plt.savefig('models/confusion_matrix.png')
    plt.close()
    
    # 保存標籤分布數據
    df['label'].value_counts().to_json('models/label_distribution.json')
    
    # 保存評估報告（JSON）
    with open('models/metrics.json', 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    print("完成！模型和相關文件已保存到 'models' 目錄")

if __name__ == '__main__':
    train_and_save_model()