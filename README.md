# IF-AutoGen v2.0

**Pharmaceutical Interview Form (IF) Automated Generation & Tuning System**  
**医薬情報適正化に向けたインタビューフォーム(IF)自動生成・チューニングシステム**  
**제약 인터뷰 폼(IF) 자동 생성 및 튜닝 시스템**

---

## 🌎 Overview / 概要 / 개요

### [English]
IF-AutoGen v2.0 is an advanced AI system designed to automate the creation and iterative optimization of pharmaceutical Interview Forms (IF) based on the **2018 IF Guideline (updated in 2019)**. Utilizing **Local LLMs** and **Multi-modal RAG**, the system extracts precise data from Common Technical Documents (CTD) while ensuring zero data leakage, making it suitable for sensitive pharmaceutical environments.

### [日本語]
IF-AutoGen v2.0は、**IF記載要領2018（2019年更新版）**に基づき、医薬品インタビューフォーム(IF)を自動生成・反復最適化する高度なAIシステムです。**ローカルLLM**と**マルチモーダルRAG**を活用し、CTD（コモン・テクニカル・ドキュメント）から正確なデータ（テキスト、表、図表）を抽出します。完全ローカル実行により機密情報の外部漏洩リスクをゼロに抑え、製薬業界の厳しいセキュリティ要件に対応します。

### [한국어]
IF-AutoGen v2.0은 **IF 기재요령 2018(2019년 개정판)**에 근거하여 제약 인터뷰 폼(IF)을 자동 생성하고 반복적으로 최적화하는 고성능 AI 시스템입니다. **로컬 LLM**과 **멀티모달 RAG**를 활용하여 CTD(공통기술문서)로부터 정확한 데이터(텍스트, 표, 도표)를 추출합니다. 모든 처리가 로컬에서 수행되므로 기밀 정보의 외부 유출 위험이 없으며, 제약 산업의 엄격한 보안 요구사항을 충족합니다.

---

## ✨ Core Features / 主要機能 / 주요 기능

1.  **Multi-modal RAG (Phase 1):**
    *   Extracts Text, Tables (Lattice/Stream), and Figures (using Vision-LLM) from CTD PDFs.
    *   Contextual chunking that preserves cross-references between text and tables.
2.  **Section-wise Generation (Phase 2):**
    *   Generates all 13 sections of the 2018 IF Standard.
    *   Localized prompting for high-precision extraction of PK/PD parameters.
3.  **Multi-metric Evaluation (Phase 3):**
    *   Evaluates quality using 8 metrics: Numerical Accuracy, Table Reproduction, BERTScore, ROUGE-L, etc.
4.  **Adaptive Tuning Loop (Phase 4):**
    *   Automatically improves prompts for low-scoring sections through iterative feedback loops.
    *   Granular section-wise tuning via `tune_section.bat`.
5.  **Monitoring Dashboard:**
    *   Real-time tracking of score improvements via Streamlit and MLflow.

---

## 🛠 Technology Stack / 技術スタック / 기술 스택

*   **Core LLM:** Ollama (Qwen2.5-Coder:7b, Llama3.1)
*   **Vision LLM:** Moondream, Qwen2-VL
*   **Vector DB:** ChromaDB
*   **Embeddings:** HuggingFace `multilingual-e5-small`
*   **Tracking:** MLflow
*   **UI:** Streamlit
*   **Morphology:** SudachiPy (Japanese Tokenization)

---

## 🚀 Getting Started / 使い方 / 사용 방법

### Prerequisites
*   Python 3.11+
*   Ollama (Local LLM server)

### Installation
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Execution
1.  **Full Pipeline (Index + Tune):**
    ```bash
    python main.py --mode full --drug_id <DRUG_ID>
    ```
2.  **Section-wise Tuning:**
    *   Run `tune_section.bat` for interactive tuning of a specific section.
3.  **Monitoring Dashboard:**
    ```bash
    streamlit run src/dashboard/app.py
    ```

---

## 🔒 Security & Privacy

This application is designed for **100% Local Execution**.
*   No cloud APIs are used (Ollama/ChromaDB/MLflow are all local).
*   No training on user-uploaded CTD documents.
*   Initial model download is the only time internet access is required.

---
© 2026 IF-AutoGen Project Team. Developed with Antigravity.
