# 垃圾郵件分類系統 (Spam Classification)

本專案為課堂作業：實作一個使用機器學習 (SVM) 的垃圾短信/郵件分類系統，並提供互動式的 Streamlit 展示頁面。

本文 README 提供專案概覽、運行步驟、檔案說明與部署建議，便於評分與展示。

## 主要特色

- ML pipeline：資料載入 → 前處理 → TF-IDF 特徵 → SVM 訓練 → 評估
- 預先計算並儲存評估資料（confusion matrix、classification report、label distribution），供前端快速載入
- 互動式展示：Streamlit 應用會讀取預計算結果並顯示圖表與單筆預測介面（無須在網頁端訓練模型）
- OpenSpec 支援：包含 `openspec/project.md` 與變更提案 `openspec/changes/add-spam-classification/`

## 專案結構（重要檔案）

```
物聯網_HW3/
├── notebooks/                              # Jupyter Notebook（實驗紀錄）
│   └── phase1_spam_classification.ipynb
├── openspec/                               # OpenSpec 文件與提案
├── src/                                    # 主要程式碼
│   ├── train_model.py       # 訓練並匯出模型與預計算評估檔案（models/）
│   ├── spam_classifier.py   # 原始腳本版的分類器實作（可當作 library）
│   └── app.py               # Streamlit 應用（讀取 models/，顯示圖表與互動預測）
├── models/                                 # 儲存訓練後的 artefacts（可放在 repo 或在部署時產生）
│   ├── spam_model.joblib
│   ├── vectorizer.joblib
│   ├── confusion_matrix.png
│   ├── label_distribution.json
│   └── metrics.json
├── requirements.txt                        # 部署依賴（Streamlit Cloud 或 CI 會用到）
└── README.md
```

## 環境與相依套件

- Python 3.8+
- 在開發時我使用的主要套件：pandas, numpy, scikit-learn, matplotlib, seaborn, streamlit, joblib, Pillow
- 已在 `requirements.txt` 列出所有必要套件，部署到 Streamlit Cloud 時會自動安裝。

## 快速上手（本地開發）

1. 取得程式碼

```powershell
git clone https://github.com/brant502/5114056015_HW3.git
cd 5114056015_HW3
```

2. 建議建立虛擬環境並安裝依賴

```powershell
python -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
pip install -r requirements.txt
```

3. 訓練模型（只需執行一次，會在 `models/` 產生必要檔案）

```powershell
python src/train_model.py
```

執行結束後會產生：
- `models/spam_model.joblib` （保存訓練好的模型）
- `models/vectorizer.joblib` （保存 TF-IDF 向量器）
- `models/confusion_matrix.png`（混淆矩陣圖）
- `models/label_distribution.json`（label 分布）
- `models/metrics.json`（classification report 的 JSON）

4. 啟動 Streamlit 應用（前端）

```powershell
streamlit run src/app.py
```

開啟瀏覽器並前往 `http://localhost:8501` 即可看到互動頁面。

## 部署到 Streamlit Cloud（建議流程）

1. 在 GitHub 上的 repository 包含 `requirements.txt`（本專案已加入）。
2. 若希望 Cloud 一啟動即可顯示圖表，請把 `models/` 下的預計算檔案也 commit 到 repo（或在部署後執行 `python src/train_model.py` 產生）。
   - 注意：二進位模型檔案會增加 repo 大小，若檔案很大，建議使用 Git LFS 或在部署環境以訓練腳本產生。  
3. 在 Streamlit Cloud 建立新應用，連結到此 GitHub repo 並部署。

## 已知注意事項與故障排除

- 若在雲端出現 `import matplotlib.pyplot as plt` 的錯誤，請確認 `app.py` 在 import 前已設定非 GUI 後端：
  - 已在程式碼中加入 `matplotlib.use('Agg')`，解決無 headless 顯示環境的問題。
- 若前端顯示找不到模型或評估檔案，會出現錯誤訊息：請先在伺服器上執行 `python src/train_model.py` 或把 `models/` 檔案放到 repo。
- Streamlit 的 `st.image(..., use_column_width=True)` 已被標記為過時，程式中已改用 `use_container_width` 建議在未來版本中更新呼叫。

## 評估摘要（從測試執行）

- 數據集大小：5,574 條短信
- 標籤分布：ham = 4,827，spam = 747
- 在一次基準訓練（SVM）上取得近似結果：
  - Accuracy: ~98%
  - spam class: precision ~1.00, recall ~0.88, f1 ~0.94

（詳細指標可參考 `models/metrics.json`）

## 建議的未來改進（短期）

- 輸出 top-k 特徵權重（TF-IDF +線性模型係數）並在前端顯示，增加可解釋性
- 嘗試不同模型（logistic regression、random forest）並比較
- 增加單元測試與 CI（例如 GitHub Actions）以驗證 pipeline

## 聯絡與引用

如果需要我幫您：把 `models/` 上傳到 repo、改用 LFS、或部署 Streamlit Cloud，告訴我您想要哪一步，我會接續處理。

---

最後更新：請以 repo 的 `main` 分支為準，README 會隨更新持續改進。
