# 垃圾郵件分類系統

這是一個使用機器學習方法（支持向量機, SVM）來分類垃圾郵件的系統。

## 專案結構

```
物聯網_HW3/
├── notebooks/
│   └── phase1_spam_classification.ipynb    # Jupyter notebook 實驗記錄
├── openspec/
│   ├── project.md                         # 專案配置和約定
│   ├── AGENTS.md                          # AI 助手工作流程說明
│   └── changes/
│       └── add-spam-classification/       # 變更提案
│           ├── proposal.md                # 提案說明
│           ├── tasks.md                   # 實施任務清單
│           └── specs/
│               └── ml-model/
│                   └── spec.md            # 模型規格說明
├── src/
│   ├── spam_classifier.py                 # 核心分類器實現
│   └── app.py                            # Streamlit 網頁應用
└── README.md                              # 本文件
```

## 功能特點

1. 數據處理
   - 自動載入和預處理郵件數據
   - 文本向量化（使用 TF-IDF）
   - 數據集切分（訓練集/測試集）

2. 模型訓練
   - 使用支持向量機 (SVM) 分類器
   - 自動化的訓練流程
   - 模型性能評估

3. 視覺化和解釋
   - 混淆矩陣視覺化
   - 分類報告（precision, recall, F1-score）
   - 標籤分布圖表

4. 互動式網頁界面
   - 實時郵件分類
   - 分類結果和置信度顯示
   - 數據集統計和視覺化

## 安裝說明

1. 安裝依賴套件：
```bash
pip install pandas numpy scikit-learn matplotlib seaborn streamlit
```

2. 克隆專案：
```bash
git clone [您的專案URL]
cd 物聯網_HW3
```

## 使用方法

1. 運行 Jupyter Notebook（用於開發和實驗）：
```bash
jupyter notebook notebooks/phase1_spam_classification.ipynb
```

2. 運行網頁應用：
```bash
streamlit run src/app.py
```

3. 直接運行分類器：
```bash
python src/spam_classifier.py
```

## 評估結果

模型在測試集上達到了以下性能：
- 整體準確率：98%
- 垃圾郵件檢測精確率：100%
- 正常郵件檢測召回率：100%

## 未來改進計劃

1. Phase 2：模型優化和特徵工程
   - 嘗試不同的特徵提取方法
   - 實驗其他機器學習算法

2. Phase 3：多模型集成
   - 實現投票或堆疊集成
   - 提高模型魯棒性

3. Phase 4：部署和實時預測
   - API 服務部署
   - 批量預測功能

4. Phase 5：效能優化和監控
   - 模型壓縮和加速
   - 預測效能監控