# IF-AutoGen v2.0 — システムアーキテクチャ文書

> **マルチモーダルRAG | テキスト・表・グラフ参照関係精密維持 | 完全ローカル実行**
> Version 2.0 | 2026年3月 | IF記載要領2018（2019年更新版）完全準拠

---

## 目次

1. [システム全体アーキテクチャ](#1-システム全体アーキテクチャ)
2. [Layer 1: CTD文書処理パイプライン](#2-layer-1-ctd文書処理パイプライン)
3. [Layer 3: マルチインデックス ChromaDB 設計](#3-layer-3-マルチインデックス-chromadb-設計)
4. [Layer 4: IFセクション特化型RAGパイプライン](#4-layer-4-ifセクション特化型ragパイプライン)
5. [Layer 5: IFセクション生成エンジン](#5-layer-5-ifセクション生成エンジン)
6. [Layer 7: 適応型チューニングループ](#6-layer-7-適応型チューニングループ)
7. [ディレクトリ構造・設定ファイル設計](#7-ディレクトリ構造設定ファイル設計)

---

## 1. システム全体アーキテクチャ

IF-AutoGen v2.0 は、CTD文書（日英混在テキスト・表・グラフ画像の複合構造）を高精度に処理し、IF記載要領2018（2019年更新版）の全13セクション（Ⅰ〜ⅩⅢ）に準拠したIF文書をローカルLLMで自動生成し、人間作成文書との一致率を反復チューニングで向上させるシステムである。

```
┌─────────────────────────────────────────────────────────────────────────┐
│                 IF-AutoGen v2.0  System Architecture                      │
│            IF記載要領2018(2019年更新版) / CTD Module 1〜5 対応             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│  LAYER 1: CTD Document Ingestion Pipeline                                 │
│    CTD PDFs (Module 2〜5)                                                 │
│     ├── Text Extractor (PyMuPDF)    → 日英混在テキスト                    │
│     ├── Table Extractor (Camelot+pdfplumber) → 構造化JSON表               │
│     ├── Figure Extractor (PyMuPDF)  → 図画像 + キャプション               │
│     ├── Vision-LLM (Qwen2-VL:7b)   → 図内容テキスト化（ローカル）         │
│     └── Cross-Reference Linker     → テキスト↔表↔図 参照リンク            │
│                          │                                                │
│  LAYER 2: Multimodal Hierarchical Chunker                                 │
│    ContextualChunk {                                                       │
│      parent_text: str (1024 tok),                                         │
│      child_texts: List[str] (256 tok each),                               │
│      linked_tables: List[TableChunk],   ← 参照表を直結                    │
│      linked_figures: List[FigureChunk], ← 参照図を直結                    │
│      metadata: {ctd_module, section, lang, page, ref_ids}                 │
│    }                                                                       │
│                          │                                                │
│  LAYER 3: Multi-Index Vector Store (ChromaDB — ローカル永続化)             │
│    ctd_{drug_id}_text   (Dense: multilingual-e5-large)                    │
│    ctd_{drug_id}_table  (Dense + BM25 hybrid)                             │
│    ctd_{drug_id}_figure (Vision embedding)                                │
│    ctd_{drug_id}_crossref (参照リンクグラフ)                               │
│                          │                                                │
│  LAYER 4: IF-Section-Aware RAG Pipeline                                   │
│    Per-Section Query Builder → IF記載要領2018セクション別クエリ生成        │
│    Hybrid Retriever          → Dense(0.6) + BM25(0.4) + RRF              │
│    Cross-Modal Expander      → テキスト検索→関連表・図を自動展開           │
│    Reranker (cross-encoder)  → Top-20 → Top-5 精度向上                   │
│    Context Assembler         → テキスト+表JSON+図キャプション統合          │
│                          │                                                │
│  LAYER 5: IF Document Generator (Ollama Local LLM)                        │
│    Section Router   → Ⅰ〜ⅩⅢ の各セクションへルーティング                 │
│    PromptBuilder    → セクション別System/User/Few-shotプロンプト組立       │
│    LLM Engine       → Qwen2.5:14b/72b (Ollama API, 完全ローカル)          │
│    PostProcessor    → 数値検証・参照番号整合チェック                        │
│    DocAssembler     → 全セクション統合 → Word文書出力 (python-docx)        │
│                          │                                                │
│  LAYER 6: Multi-Metric Quality Evaluator (8メトリクス)                    │
│    TextSim:       ROUGE-L + BERTScore(ja)          [15%]                  │
│    NumericalAcc:  数値・単位一致率                   [25%]                 │
│    TableRepr:     表構造一致率                       [15%]                 │
│    FigRefMatch:   図表番号参照整合性                  [10%]                │
│    SectionCov:    Ⅰ〜ⅩⅢ全セクション準拠率            [10%]                │
│    SemanticSim:   multilingual BERTScore             [15%]                │
│    TermCov:       医薬専門用語カバレッジ               [5%]                │
│    CitationMatch: 文献一致率                         [5%]                 │
│    CoverComplete: 表紙20項目完備率                   [5%]                 │
│                          │                                                │
│  LAYER 7: Adaptive Tuning Loop Controller                                 │
│    Config: max_loops=N, target_score=T                                    │
│    PromptOptimizer   → 低スコアセクション別プロンプト強化                  │
│    RAGParamOptimizer → Top-K・チャンクサイズ・BM25重み調整                │
│    CrossRefTuner     → 表・図参照精度の専用チューニング（v2新機能）         │
│                          │                                                │
│  LAYER 8: Monitoring Dashboard (Streamlit + MLflow — ローカル)            │
│    Score Timeline | Section Heatmap | Diff Viewer                         │
│    Parameter History | Drug Comparison | Cross-Ref Quality Panel          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Layer 1: CTD文書処理パイプライン

### 2.1 マルチモーダル文書解析アーキテクチャ

```
CTD PDF Input
    │
    ├──▶ [Text Parser: PyMuPDF]
    │       ├── ページ単位テキストブロック抽出
    │       ├── フォント情報（見出し判定: サイズ/Bold）
    │       ├── 段落境界検出（行間距離分析）
    │       ├── 言語判定（日本語/英語ブロック分類）
    │       └── セクション番号検出（2.7.2.1等の正規表現マッチ）
    │
    ├──▶ [Table Parser: Camelot + pdfplumber]
    │       ├── 格子型表 (lattice mode): 罫線検出→セル抽出
    │       ├── 非格子型表 (stream mode): 空白列幅分析→セル抽出
    │       ├── ヘッダ行自動判定（フォント・位置分析）
    │       ├── 結合セル検出・展開
    │       ├── 脚注抽出・メイン表への関連付け
    │       └── 表番号/タイトル抽出（Table N / 表N パターン）
    │
    ├──▶ [Figure Parser: PyMuPDF画像抽出]
    │       ├── 図画像（PNG/JPEG）抽出・番号付け
    │       ├── キャプション抽出（Figure N / 図N 直後テキスト）
    │       ├── 図種別推定（折れ線/棒/散布図/化学構造/フローチャート）
    │       └── 図番号テキスト参照との対応付け準備
    │
    ├──▶ [Vision-LLM: Qwen2-VL:7b (Ollama ローカル)]
    │       ├── グラフ図のX/Y軸ラベル・単位の抽出
    │       ├── 凡例（Legend）テキストの抽出
    │       ├── PKグラフ: データ点の近似数値読取り
    │       ├── 表グラフ: 数値ラベルの抽出
    │       └── 化学構造図: 構造の文字化（SMILES補完）
    │
    └──▶ [Cross-Reference Linker]
            ├── 「表N」「Table N」言及 → 対応表チャンクへのリンク
            ├── 「図N」「Figure N」言及 → 対応図チャンクへのリンク
            ├── 双方向リンクグラフ構築
            └── 孤立参照（対応表/図なし）の検出・ログ
```

### 2.2 コンテキスト保持チャンク構造

```python
@dataclass
class ContextualChunk:
    # ─── テキスト本体 ───────────────────────────────────────
    chunk_id:        str          # 一意ID
    parent_text:     str          # 親チャンク (≈1024 tokens)
    child_texts:     List[str]    # 子チャンク (≈256 tokens each)

    # ─── 参照リンク（★核心機能）────────────────────────────
    linked_tables:   List[TableChunk]   # 参照される表を直結
    linked_figures:  List[FigureChunk]  # 参照される図を直結
    ref_ids:         List[str]          # 「表1」「図2」等のID

    # ─── メタデータ ────────────────────────────────────────
    ctd_module:      str    # "Module_2.7.2"
    ctd_section:     str    # "2.7.2.1"
    if_section_hint: str    # "section_VII" (対応IFセクション)
    language:        str    # "ja" / "en" / "mixed"
    page_num:        int
    doc_source:      str    # CTDファイル名

@dataclass
class TableChunk:
    table_id:    str         # "Table_1"
    title:       str         # 表タイトル
    headers:     List[str]   # ヘッダ行
    rows:        List[List[str]]  # データ行
    footnotes:   List[str]   # 脚注（略語定義等）
    json_repr:   str         # JSON文字列（LLM入力用）
    ctd_section: str
    page_num:    int

@dataclass
class FigureChunk:
    figure_id:    str        # "Figure_1"
    caption:      str        # キャプション全文
    figure_type:  str        # "pk_profile" / "km_curve" / "bar" / "structure"
    vision_text:  str        # Vision-LLMが抽出したテキスト（軸ラベル・数値）
    image_path:   str        # 画像ファイルパス
    ctd_section:  str
    page_num:     int
```

### 2.3 CTDモジュール別処理戦略

| CTDモジュール | テキスト言語 | 表密度 | グラフ密度 | 特殊処理 |
|-------------|------------|--------|----------|---------|
| Module 2.3 (品質概括) | 日英混在 | 中 | 低 | 製剤技術用語辞書適用 |
| Module 2.6.2 (薬理概括) | 日英混在 | 高 | 高（用量反応曲線） | グラフ軸ラベル抽出必須 |
| Module 2.7.1 (BA試験概要) | 英語主体 | 高 | 高（PKプロファイル） | **★PKグラフ+パラメータ表の対応付けが最重要** |
| Module 2.7.2 (臨床薬理) | 英語主体 | 非常に高 | 高 | PKパラメータ表の完全抽出。数値単位統一 |
| Module 2.7.3 (有効性) | 英語主体 | 非常に高 | 高（KM曲線・Forest plot） | 試験結果統合表。複数試験の統合処理 |
| Module 2.7.4 (安全性) | 英語主体 | 非常に高 | 中 | SOC別副作用発現率表の完全抽出 |
| Module 3.2.P (製剤) | 日英混在 | 高 | 中（安定性曲線） | 安定性グラフと数値表の対応維持 |
| Module 4.2.1 (薬理試験) | 英語主体 | 高 | 高（用量反応） | 非臨床グラフのED50・IC50との対応 |
| Module 4.2.3 (毒性試験) | 英語主体 | 非常に高 | 低 | 毒性所見表・NOAEL一覧表の完全抽出 |
| Module 5.3.1 (BA報告書) | 英語主体 | 非常に高 | 高 | 個票レベルPKデータ表。統計解析結果表 |
| Module 5.3.5 (有効性・安全性) | 英語主体 | 非常に高 | 高 | 症例データ表・CONSORT図との対応 |

---

## 3. Layer 3: マルチインデックス ChromaDB 設計

```
ChromaDB Persistent Storage: ./data/vectordb/{drug_id}/

Collection 1: ctd_{drug_id}_text
  Embedding: multilingual-e5-large (1024-dim)
  Distance:  cosine
  Metadata:  {ctd_module, section, lang, page, chunk_type(parent/child),
              parent_id, linked_table_ids[], linked_figure_ids[]}
  用途: メインテキスト Dense 検索

Collection 2: ctd_{drug_id}_table
  Embedding: multilingual-e5-large (table title + headers + first row)
  Distance:  cosine
  Metadata:  {table_id, title, ctd_module, section, page, footnote_count}
  Document:  JSON形式の表全体
  用途: 表内容の検索・完全取得

Collection 3: ctd_{drug_id}_figure
  Embedding: multilingual-e5-large (caption + vision_text)
  Distance:  cosine
  Metadata:  {figure_id, figure_type, ctd_module, section, page}
  Document:  caption + vision_text (Vision-LLM抽出テキスト)
  用途: 図説明の検索・キャプション取得

Collection 4: ctd_{drug_id}_crossref
  構造: 参照リンクグラフ（テキストID → 表ID[], 図ID[]）
  用途: クロスモーダル展開（テキスト検索後に関連表・図を自動取得）
```

### ChromaDB インデックスパラメータ

| パラメータ | テキストCollection | 表Collection | チューニング範囲 |
|-----------|------------------|------------|----------------|
| チャンクサイズ（親） | 1024 tokens | 表全体 | 512〜2048 |
| チャンクサイズ（子） | 256 tokens | — | 128〜512 |
| hnsw:space | cosine | cosine | 固定 |
| hnsw:M（接続数） | 16 | 16 | 8〜32 |
| hnsw:ef_construction | 200 | 200 | 100〜400 |

---

## 4. Layer 4: IFセクション特化型RAGパイプライン

### 4.1 セクション別クエリ戦略

| IFセクション | クエリテンプレート | 優先CTDコレクション | クロスモーダル展開 |
|------------|-----------------|------------------|----------------|
| Ⅰ. 概要 | 開発の経緯 {drug} 疾患背景 治療学的特性 | text(2.5) | 表あり→試験概要表 |
| Ⅱ. 名称 | {drug} 一般名 分子式 分子量 化学名 | text(3.2.S.1) | なし |
| Ⅲ. 有効成分 | {drug} 物理化学的性質 溶解性 安定性 pKa | text+table | ★溶解性表・安定性表を自動展開 |
| Ⅳ. 製剤 | {drug} 剤形 製剤組成 安定性試験 溶出 | text+table+figure | ★安定性グラフ+数値表の同時取得 |
| Ⅴ. 臨床成績 | {drug} 有効性 主要評価項目 試験結果 p値 | text+table+figure | ★KM曲線+統計表の同時取得 |
| Ⅵ. 薬効薬理 | {drug} 薬理作用 IC50 作用機序 選択性 | text+table+figure | ★用量反応グラフ+IC50表 |
| **Ⅶ. PK（最重要）** | {drug} Cmax AUC t1/2 pharmacokinetics | text+table+figure | **★PKグラフ+パラメータ表の完全取得** |
| Ⅷ. 安全性 | {drug} 副作用 有害事象 SOC 発現率 | text+table | ★SOC別発現率表の完全取得 |
| Ⅸ. 非臨床 | {drug} 毒性 NOAEL 遺伝毒性 がん原性 | text+table | ★毒性試験NOAEL一覧表 |
| ⅩⅢ. 備考（簡易懸濁） | {drug} 簡易懸濁 粉砕 安定性 懸濁後 | text+table+figure | 懸濁後安定性グラフ |

### 4.2 クロスモーダル展開プロセス

```python
def cross_modal_expand(text_results: List[ContextualChunk],
                       table_col, figure_col,
                       max_tables=3, max_figures=2) -> RetrievalContext:
    """
    テキスト検索結果から参照される表・図を自動展開する
    「表1参照」とテキストにあれば表1の完全データを取得できる
    """
    context = RetrievalContext()

    for chunk in text_results:
        context.add_text(chunk.parent_text)

        # 参照表を展開（直結リンクから優先）
        for table_id in chunk.linked_table_ids[:max_tables]:
            table_data = table_col.get(ids=[table_id])
            if table_data:
                context.add_table(table_id, table_data)  # JSON全体

        # 参照図を展開（Vision-LLM抽出テキスト+キャプションを追加）
        for fig_id in chunk.linked_figure_ids[:max_figures]:
            fig_data = figure_col.get(ids=[fig_id])
            if fig_data:
                context.add_figure(fig_id,
                    caption=fig_data.caption,
                    vision_text=fig_data.vision_text)

    return context

# 最終コンテキスト構造（LLMへの入力）:
"""
## 参照テキスト
[Module 2.7.2.1 / 2.7.2] テキスト本体...

## Table_1
{"headers": ["Dose","n","Cmax (ng/mL)","Tmax (h)","AUC0-inf (ng·h/mL)","t1/2 (h)"],
 "rows": [["10 mg","8","245±42","2.1 (1.5-3.0)","1234±189","12.3±1.8"],
          ["20 mg","8","498±67","2.0 (1.5-3.0)","2456±312","12.5±2.1"]],
 "footnotes": ["Mean±SD; Tmax: median (range)"]}

## Figure_1 (グラフ情報)
Caption: 単回経口投与後の平均血漿中薬物濃度-時間プロファイル
Content: X-axis: Time (h), 0-48; Y-axis: Concentration (ng/mL), 0-600
         Lines: 10mg (■), 20mg (●); peak at ~2h, decline to baseline at 48h
"""
```

---

## 5. Layer 5: IFセクション生成エンジン

### 5.1 セクション別プロンプト設計（Ⅶ. 薬物動態の例）

```yaml
# prompts/base/section_VII.yaml

system: |
  あなたはIF記載要領2018（2019年更新版）に完全準拠した
  医薬品インタビューフォーム作成専門家です。
  【絶対遵守ルール】
  1. 数値は参照CTDデータから1桁も変えずに転記すること
  2. 表が参照されている場合は「表N」の形式で本文中に明示すること
  3. 図が参照されている場合は「図N」の形式で本文中に明示すること
  4. 単位は元データの単位を使用し、独断で変換しないこと（ng/mL→μg/mL等）
  5. 平均値±SDの形式を維持すること
  6. 統計記号（p値・信頼区間）は元データの表記を維持すること
  7. 文体は学術文体（〜である、〜を示した、〜と考えられる）を使用すること
  8. IF記載要領2018 Ⅶ節に規定された全11サブ項目を必ず含めること

user_template: |
  以下のCTD参照データを基に、医薬品「{drug_name}」の
  インタビューフォーム「Ⅶ. 薬物動態に関する項目」を作成してください。

  {context_text}
  {context_tables}
  {context_figures}

  ## 記載要領2018 Ⅶ節 必須記載11項目
  以下の全11項目を順番に記載すること：
  1. 血中濃度の推移（単回・反復、図N参照を含む）
  2. 薬物速度論的パラメータ（Cmax・AUC・t1/2・Tmax・CL/F・Vd/F、表N参照を含む）
  3. 母集団（ポピュレーション）解析（実施された場合）
  4. 吸収（バイオアベイラビリティ・食事影響）
  5. 分布（タンパク結合率・B/P比・Vd）
  6. 代謝（代謝経路・関与CYP・代謝物・図N参照を含む）
  7. 排泄（排泄経路・尿中排泄率・CLr）
  8. トランスポーターに関する情報（P-gp・BCRP・OATP等）
  9. 透析等による除去率
  10. 特定の背景を有する患者（腎・肝障害・高齢者・小児・遺伝子多型）
  11. その他（DDI試験結果があれば記載）
```

### 5.2 数値・表・参照整合ポストプロセッサ

```python
class PostProcessor:
    """生成テキストの数値・表番号・図番号の整合性を検証・修正する"""

    def validate_and_fix(self, generated_text: str) -> Tuple[str, List[str]]:
        issues = []

        # 1. 重要数値（PKパラメータ等）が正確に転記されているか確認
        ctx_nums = self._extract_numbers_with_units(self.context.all_text)
        gen_nums = self._extract_numbers_with_units(generated_text)
        for num, unit in ctx_nums:
            if num not in gen_nums and self._is_critical_pk_value(num, unit):
                issues.append(f"重要数値未転記: {num} {unit}")

        # 2. 図表番号参照チェック
        ctx_refs = set(re.findall(r"(?:表|図|Table|Figure)\s*\d+", self.context.all_text))
        gen_refs = set(re.findall(r"(?:表|図|Table|Figure)\s*\d+", generated_text))
        missing_refs = ctx_refs - gen_refs
        if missing_refs:
            issues.append(f"未参照の図表: {missing_refs}")

        return generated_text, issues
```

---

## 6. Layer 7: 適応型チューニングループ

### 6.1 チューニングループ制御フロー

```
Adaptive Tuning Loop Controller

Input: config.yaml (max_loops: N, target_score: T)
       reference_if: Dict[section → human_text]

LOOP i (i = 1 ... N):

  ① RAG実行 → コンテキスト生成（テキスト+表+図）
  ② LLM生成 → IF全13セクション逐次生成
  ③ PostProcess → 数値・参照整合チェック・自動修正
  ④ 評価 → 9メトリクス計算（数値一致25%・表再現15%等）
  ⑤ ログ → MLflow記録（スコア・パラメータ・生成IF保存）
  ⑥ 終了判定 → composite ≥ T  OR  i ≥ N  OR  早期停止?
  ⑦ チューニング（3種類の最適化器を並行実行）:
     a. PromptOptimizer
        低スコアセクション特定 → セクション別プロンプト強化
        Few-shot例を最良生成物から更新
     b. RAGParamOptimizer
        数値一致率低下 → Top-K増加・チャンクサイズ調整
        表再現率低下 → 表コレクション重み増加
        図参照不一致 → クロスモーダル展開数増加
     c. CrossRefTuner（★v2.0新機能）
        図表参照整合スコア < 0.6 → 参照リンク再構築
        孤立参照の手動マッピング候補を提示

Output: best_if_document + tuning_history + final_scores
```

### 6.2 チューニング対象パラメータ一覧

| カテゴリ | チューニング対象パラメータ | 最適化手法 | スコア改善期待 |
|---------|----------------------|---------|-------------|
| Prompt | セクション別システムプロンプト内容 | スコア依存強化（低スコアセクションに指示追加） | BERTScore +5〜12% |
| Prompt | Few-shot例の品質・量（0〜3例） | 最良生成物からの自動更新 | 全体 +5〜15% |
| RAG | Top-K（retrieval前） | Recall分析に基づくグリッドサーチ（10〜30） | 数値一致 +3〜8% |
| RAG | Top-K（rerank後） | Precision分析（3〜10） | 精度 +3〜7% |
| RAG | BM25重み（0.2〜0.6） | キーワード一致率に基づく調整 | 専門用語 +5〜10% |
| RAG | 親チャンクサイズ（512〜2048 tok） | 文脈長分析 | ROUGE-L +3〜7% |
| RAG | クロスモーダル展開数（表:1〜5, 図:1〜3） | 表再現・図参照スコアに基づく調整 | 表再現 +8〜20% |
| Eval | 評価重み（9メトリクス） | スコア分布分析による動的調整 | Composite最適化 |
| LLM | Temperature（0.0〜0.3） | 一貫性スコア分析 | 生成一貫性改善 |

---

## 7. ディレクトリ構造・設定ファイル設計

### 7.1 ディレクトリ構造

```
if-autogen-v2/
├── config.yaml                      # システム全設定
├── main.py                          # エントリーポイント
│
├── data/
│   ├── raw/{drug_id}/
│   │   ├── ctd/                     # CTD PDFs (Module 2〜5)
│   │   │   ├── M2_7_2_clinpharm.pdf
│   │   │   ├── M2_7_3_efficacy.pdf
│   │   │   ├── M2_7_4_safety.pdf
│   │   │   ├── M4_2_3_tox_studies.pdf
│   │   │   └── M5_3_5_clin_studies.pdf
│   │   └── if_human.docx            # 人間作成IF（参照文書）
│   ├── processed/{drug_id}/
│   │   ├── text_chunks.json
│   │   ├── table_chunks.json
│   │   ├── figure_chunks.json
│   │   └── crossref_graph.json
│   └── vectordb/{drug_id}/          # ChromaDB永続化（4コレクション）
│
├── models/                          # ローカルモデルキャッシュ
│
├── prompts/
│   ├── base/                        # 初期プロンプトテンプレート（全13セクション）
│   │   ├── section_I_overview.yaml
│   │   ├── section_VII_pk.yaml      # 最重要・最詳細
│   │   ├── section_VIII_safety.yaml
│   │   └── section_XIII_remarks.yaml  # ★2019年更新版新設
│   └── tuned/                       # チューニング済みプロンプト
│
├── src/
│   ├── processor/
│   │   ├── text_extractor.py        # PyMuPDF テキスト抽出
│   │   ├── table_extractor.py       # Camelot+pdfplumber 表抽出
│   │   ├── figure_extractor.py      # PyMuPDF 図抽出
│   │   ├── vision_analyzer.py       # Qwen2-VL グラフ解析（ローカル）
│   │   ├── crossref_linker.py       # テキスト↔表↔図参照リンク構築
│   │   └── chunker.py               # 階層型コンテキスト保持チャンカー
│   ├── indexer/
│   │   ├── multi_index_builder.py   # 4種ChromaDBコレクション構築
│   │   └── bm25_index.py            # BM25インデックス（SudachiPy）
│   ├── rag/
│   │   ├── retriever.py             # Hybrid Retrieval + Reranker
│   │   ├── cross_modal_expander.py  # クロスモーダル展開
│   │   └── context_assembler.py     # テキスト+表+図 コンテキスト組立
│   ├── generator/
│   │   ├── section_generator.py     # セクション別IF生成
│   │   ├── prompt_builder.py        # セクション別プロンプト組立
│   │   ├── post_processor.py        # 数値・参照整合チェック
│   │   └── doc_assembler.py         # IF全文書組立・Word出力
│   ├── evaluator/
│   │   ├── text_evaluator.py        # ROUGE + BERTScore
│   │   ├── numerical_evaluator.py   # 数値・単位一致率
│   │   ├── table_evaluator.py       # 表構造再現率
│   │   ├── crossref_evaluator.py    # 図表参照整合性
│   │   └── composite_evaluator.py   # 複合スコア計算
│   ├── tuner/
│   │   ├── tuning_loop.py           # メインチューニングループ
│   │   ├── prompt_optimizer.py      # プロンプト自動改善
│   │   ├── rag_optimizer.py         # RAGパラメータ最適化
│   │   └── crossref_tuner.py        # 参照整合チューニング（v2新機能）
│   └── dashboard/
│       └── app.py                   # Streamlit監視ダッシュボード
│
├── outputs/{drug_id}/
│   ├── generated/                   # ループ別生成IF (best_loopN.json)
│   └── evaluations/                 # ループ別スコアJSON
├── mlruns/                          # MLflow実験記録（ローカル）
└── tests/                           # Step別テストプログラム（8ファイル）
```

### 7.2 config.yaml 完全版

```yaml
# IF-AutoGen v2.0 設定ファイル

system:
  drug_id: "drug_A"            # 処理対象医薬品ID
  drug_name: "○○錠10mg"       # 医薬品名（プロンプトに使用）
  mode: "full"                 # full | index | generate | evaluate | tune
  if_standard: "2018_2019"     # IF記載要領バージョン（固定）

llm:
  provider: "ollama"           # 完全ローカル実行
  model: "qwen2.5:14b"         # 主力モデル
  vision_model: "qwen2-vl:7b"  # 図解析専用（ローカル）
  temperature: 0.1
  max_tokens_per_section: 4096
  num_ctx: 8192
  ollama_host: "http://localhost:11434"  # ローカルのみ

rag:
  embedding_model: "intfloat/multilingual-e5-large"  # ローカルキャッシュ
  reranker_model: "cross-encoder/mmarco-mMiniLMv2-L12-H384"
  chunk_size_parent: 1024
  chunk_size_child: 256
  top_k_retrieve: 20
  top_k_rerank: 5
  bm25_weight: 0.4
  dense_weight: 0.6
  cross_modal:
    max_linked_tables: 3       # テキストチャンクあたりの展開表数上限
    max_linked_figures: 2      # テキストチャンクあたりの展開図数上限
    enable_vision_llm: true    # Vision-LLMによる図解析ON/OFF

tuning:
  max_loops: 15                # 最大チューニングループ回数
  target_score: 0.85           # ターゲット複合スコア（0.0〜1.0）
  target_metric: "composite"   # 判定基準
  early_stop_patience: 3       # N回連続改善なし→早期終了
  enable_prompt_tuning: true
  enable_rag_tuning: true
  enable_crossref_tuning: true  # v2.0新機能

evaluation:
  weights:
    text_similarity:     0.10
    semantic_similarity: 0.15
    numerical_accuracy:  0.25   # 最重要 - 医学的正確性
    table_reproduction:  0.15
    figure_ref_match:    0.10
    section_coverage:    0.10
    term_coverage:       0.05
    citation_match:      0.05
    cover_completeness:  0.05

monitoring:
  mlflow_uri: "./mlruns"        # ローカルMLflowサーバ
  dashboard_port: 8501
  log_level: "INFO"
```

> **重要**: すべての処理はローカルPC上で完結する。製薬企業の機密CTD文書が外部に漏洩するリスクはゼロ。
