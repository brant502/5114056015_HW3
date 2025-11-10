import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(url):
    """載入並預處理數據"""
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
    svm.fit(X_train, y_train)
    return svm

def evaluate_model(model, X_test, y_test):
    """評估模型"""
    y_pred = model.predict(X_test)
    print('分類報告:')
    print(classification_report(y_test, y_pred))
    
    # 繪製混淆矩陣
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('混淆矩陣')
    plt.ylabel('真實標籤')
    plt.xlabel('預測標籤')
    plt.savefig('confusion_matrix.png')
    plt.close()

def main():
    # 載入數據
    url = 'https://raw.githubusercontent.com/PacktPublishing/Hands-On-Artificial-Intelligence-for-Cybersecurity/refs/heads/master/Chapter03/datasets/sms_spam_no_header.csv'
    df = load_data(url)
    print('數據集大小:', df.shape)
    print('\n標籤分布:')
    print(df['label'].value_counts())

    # 特徵工程
    X, y, vectorizer = prepare_features(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 訓練模型
    print('\n開始訓練模型...')
    model = train_model(X_train, y_train)

    # 評估模型
    print('\n模型評估結果:')
    evaluate_model(model, X_test, y_test)

    return model, vectorizer

if __name__ == '__main__':
    main()