# Project Context

## Purpose
建立一個垃圾郵件分類系統，使用機器學習方法來自動識別和分類垃圾郵件。此專案將分階段實施，從基礎模型開始，逐步改進分類效能。

## Tech Stack
- Python (主要開發語言)
- Scikit-learn (機器學習框架)
- Pandas (數據處理)
- NumPy (數值計算)
- Jupyter Notebook (開發環境)

## Project Conventions

### Code Style
- 使用 PEP 8 Python 代碼規範
- 所有函數和類別需要包含文檔字符串(docstring)
- 變數命名採用 snake_case
- 類別命名採用 PascalCase

### Architecture Patterns
- 模組化設計，將數據處理、模型訓練、評估分開
- 使用 Pipeline 模式處理數據預處理和模型訓練
- 採用工廠模式管理不同的機器學習模型

### Testing Strategy
- 使用 K-fold 交叉驗證評估模型
- 採用混淆矩陣評估分類效果
- 使用 precision, recall, F1-score 作為主要評估指標

### Git Workflow
- 主分支: master
- 功能分支命名: feature/phase-N-description
- 每個 phase 完成後合併回主分支

## Domain Context
- 專案重點在垃圾郵件分類
- 使用監督式學習方法
- 從簡單模型開始，逐步改進
- 需要處理文本數據預處理和特徵工程

## Important Constraints
- 需要保證模型的實時性能
- 要考慮模型的可解釋性
- 需要處理文本數據的多語言問題
- 考慮內存使用效率

## External Dependencies
- 數據來源: GitHub 上的 SMS Spam Dataset
- Python 3.8+
- Scikit-learn 1.0+
- Pandas, NumPy 等科學計算庫
