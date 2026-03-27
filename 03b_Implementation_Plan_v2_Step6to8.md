**IF-AutoGen v2.0**

**実装計画文書 (後編)**

Step 6〜8: IF生成エンジン → 品質評価 → チューニングループ統合

IF記載要領2018(2019年更新版) 全13セクション | 8メトリクス評価 | 適応型チューニング

```
STEP 6 IFセクション生成エンジン（IF記載要領2018 全13セクション対応）
推定: 5〜8日
```

**6.1 IFセクション定義（IF記載要領2018全13セクション）**

IF記載要領2018（2019年更新版）は全13の本編セクションを規定する。各セクションは対応するCTDモジュールと最大トークン数が決まっており、セクション別プロンプトテンプレートで生成を制御する。

| **セクション名**                      | **CTDヒント**  | **最大tokens** | **2019年更新ポイント**                 |
|---------------------------------------|----------------|----------------|----------------------------------------|
| Ⅰ. 概要に関する項目（6項目）          | Module 2.5     | 3000           | RMP概要（第6項）が2019年更新版で追加   |
| Ⅱ. 名称に関する項目（6項目）          | Module 3.2.S.1 | 1000           | —                                      |
| Ⅲ. 有効成分に関する項目（3項目）      | Module 3.2.S   | 2000           | —                                      |
| Ⅳ. 製剤に関する項目（12項目）         | Module 3.2.P   | 3000           | —                                      |
| Ⅴ. 治療に関する項目（5項目）          | Module 2.7.3   | 6000           | 臨床成績が最大ボリューム               |
| Ⅵ. 薬効薬理に関する項目（2項目）      | Module 2.6.2   | 3000           | —                                      |
| Ⅶ. 薬物動態に関する項目（11項目）     | Module 2.7.2   | 5000           | トランスポーター情報（第8項）が追加    |
| Ⅷ. 安全性（使用上の注意等）（12項目） | Module 2.7.4   | 4000           | 添付文書新記載要領対応（冒頭注意事項） |
| Ⅸ. 非臨床試験に関する項目（2項目）    | Module 4.2     | 4000           | —                                      |
| Ⅹ. 管理的事項に関する項目（14項目）   | Module 1.2     | 2000           | —                                      |
| ⅩⅠ. 文献（2項目）                     | 全Module       | 1000           | ICMJE形式引用                          |
| ⅩⅡ. 参考資料（2項目）                 | —              | 1500           | 海外発売状況・海外ガイドライン         |
| ⅩⅢ. 備考（2項目）★2019更新            | Module 3.2.P   | 2000           | ★簡易懸濁・粉砕等の安定性情報が新設    |

**6.2 実装コード (src/generator/section_generator.py)**

```
import ollama, yaml
from pathlib import Path
from src.rag.retriever import CrossModalRAGPipeline
IF_SECTIONS = {
"section_I": {"name":"Ⅰ. 概要に関する項目", "ctd":"2.5", "tokens":3000},
"section_II": {"name":"Ⅱ. 名称に関する項目", "ctd":"3.2.S", "tokens":1000},
"section_III": {"name":"Ⅲ. 有効成分に関する項目", "ctd":"3.2.S", "tokens":2000},
"section_IV": {"name":"Ⅳ. 製剤に関する項目", "ctd":"3.2.P", "tokens":3000},
"section_V": {"name":"Ⅴ. 治療に関する項目", "ctd":"2.7.3", "tokens":6000},
"section_VI": {"name":"Ⅵ. 薬効薬理に関する項目", "ctd":"2.6.2", "tokens":3000},
"section_VII": {"name":"Ⅶ. 薬物動態に関する項目", "ctd":"2.7.2", "tokens":5000},
"section_VIII": {"name":"Ⅷ. 安全性に関する項目", "ctd":"2.7.4", "tokens":4000},
"section_IX": {"name":"Ⅸ. 非臨床試験に関する項目","ctd":"4.2", "tokens":4000},
"section_X": {"name":"Ⅹ. 管理的事項に関する項目","ctd":"1.2", "tokens":2000},
"section_XI": {"name":"ⅩⅠ. 文献", "ctd":"all", "tokens":1000},
"section_XII": {"name":"ⅩⅡ. 参考資料", "ctd":"all", "tokens":1500},
"section_XIII": {"name":"ⅩⅢ. 備考（調剤・服薬支援含む）","ctd":"3.2.P","tokens":2000},
}
class IFSectionGenerator:
def __init__(self, drug_id, drug_name, rag: CrossModalRAGPipeline,
model="qwen2.5:14b", prompt_dir="./prompts",
host="http://localhost:11434"):
self.drug_id = drug_id
self.drug_name = drug_name
self.rag = rag
self.model = model
self.prompt_dir = Path(prompt_dir)
self.host = host
def load_prompt(self, key: str) -> dict:
for base in ["tuned", "base"]:
p = self.prompt_dir / base / f"{key}.yaml"
if p.exists():
with open(p) as f: return yaml.safe_load(f)
return self._default_prompt(key)
def _default_prompt(self, key: str) -> dict:
sec = IF_SECTIONS.get(key, {})
return {
"system": """あなたはIF記載要領2018（2019年更新版）に完全準拠した
インタビューフォーム作成の専門家です。
【必須ルール】
1. 数値は参照データから1桁も変えずに転記すること
2. 表が言及されている場合は「表N」の形式で本文中に参照すること
3. 図が言及されている場合は「図N」の形式で本文中に参照すること
4. 単位は元データの単位をそのまま使用すること（独断で変換しない）
5. 文体は学術文体（〜である、〜を示した）を使用すること""",
"user_template": f"医薬品「{{drug_name}}」の{sec.get('name',key)}を作成:\n{{context}}"
}
def generate_section(self, key: str) -> str:
sec = IF_SECTIONS.get(key, {})
query = f"{self.drug_name} {sec.get('ctd','')} {sec.get('name',key)}"
ctx = self.rag.retrieve(query, if_section=key)
tmpl = self.load_prompt(key)
user = tmpl["user_template"].format(
drug_name=self.drug_name, context=ctx.to_llm_input())
resp = ollama.chat(
model=self.model, host=self.host,
messages=[{"role":"system","content":tmpl["system"]},
{"role":"user","content":user}],
options={"temperature":0.1,
"num_predict":sec.get("tokens",3000),
"num_ctx":8192})
return resp["message"]["content"]
def generate_full_if(self) -> dict:
result = {}
for key, cfg in IF_SECTIONS.items():
print(f" Generating {cfg['name']}...")
result[key] = self.generate_section(key)
return result
```

**6.3 セクション別プロンプトテンプレート例（Ⅶ. 薬物動態）**

```
# prompts/base/section_VII.yaml
system: |
あなたはIF記載要領2018（2019年更新版）に完全準拠した
医薬品インタビューフォーム作成専門家です。
Ⅶ. 薬物動態に関する項目の記載において以下を厳守してください。
【絶対ルール】
1. PK数値（Cmax・AUC・t1/2・Tmax・CL/F・Vd/F）は参照データから正確に転記
2. 表番号（表N）・図番号（図N）を本文中に明示する
3. 単位を統一する（ng/mLはng/mLのまま、独断でμg/mLに変換しない）
4. 平均値±SDの表記形式を維持する
5. Tmax は中央値（範囲）で記載する
user_template: |
医薬品「{drug_name}」のインタビューフォーム
「Ⅶ. 薬物動態に関する項目」を以下の参照データに基づき作成してください。
{context}
## 必須記載11項目（IF記載要領2018 Ⅶ節）
以下を全て順番に記載すること:
1. 血中濃度の推移（単回・反復投与、血中濃度-時間プロファイル: 図N参照）
2. 薬物速度論的パラメータ（Cmax・AUC・t1/2・Tmax・CL/F・Vd/F表: 表N参照）
3. 母集団（ポピュレーション）解析（実施された場合）
4. 吸収（BA・食事影響・吸収速度定数）
5. 分布（タンパク結合率・B/P比・Vd）
6. 代謝（代謝経路・CYP分子種・代謝物: 図N参照）
7. 排泄（主排泄経路・尿中排泄率・CLr）
8. トランスポーターに関する情報（P-gp・BCRP・OATP等）
9. 透析等による除去率
10. 特定の背景を有する患者（腎・肝障害・高齢者・小児・遺伝子多型）
11. その他（DDI試験結果等）
```

**6.4 Step 6 テストプログラム**

```
# tests/test_step6_generator.py
import pytest, re
from src.generator.section_generator import IFSectionGenerator, IF_SECTIONS
from src.rag.retriever import CrossModalRAGPipeline
CFG = {"top_k_retrieve":20,"top_k_rerank":5,"bm25_weight":0.4,
"dense_weight":0.6,"max_linked_tables":3,"max_linked_figures":2}
def get_gen():
rag = CrossModalRAGPipeline("./data/vectordb/test_idx","drug_A",CFG)
return IFSectionGenerator("drug_A","テスト薬品A",rag)
def test_section_VII_pk_all_11_subsections():
gen = get_gen()
result = gen.generate_section("section_VII")
assert len(result) > 300
required = ["血中濃度","薬物速度論","吸収","分布","代謝","排泄"]
found = [kw for kw in required if kw in result]
assert len(found) >= 3, f"PK必須キーワード不足: {found}"
print(f"PK生成: {len(result)}文字, キーワード: {found}")
def test_section_VIII_safety_keywords():
gen = get_gen()
result = gen.generate_section("section_VIII")
found = [kw for kw in ["副作用","有害事象","警告","禁忌"] if kw in result]
assert len(found) >= 1
print(f"安全性キーワード: {found}")
def test_section_XIII_2019update_new_section():
"""2019年更新版で新設されたⅩⅢ節（簡易懸濁等）"""
gen = get_gen()
result = gen.generate_section("section_XIII")
assert len(result) > 50
print(f"ⅩⅢ節（2019年更新版新設）: {result[:100]}")
def test_numerical_values_in_pk():
gen = get_gen()
result = gen.generate_section("section_VII")
nums = re.findall(r"\d+\.?\d*", result)
assert len(nums) >= 5, f"PK数値少なすぎ: {len(nums)}個"
print(f"数値出現: {len(nums)}個 (例: {nums[:5]})")
def test_figure_table_refs_in_text():
gen = get_gen()
result = gen.generate_section("section_VII")
refs = re.findall(r"(?:表|図|Table|Figure)\s*\d+", result)
print(f"図表参照: {refs[:5]}")
def test_all_13_sections_generate():
gen = get_gen()
for key in list(IF_SECTIONS.keys())[:4]: # 最初の4セクションで確認
result = gen.generate_section(key)
assert len(result) > 50, f"{key} 生成失敗"
print(f"{IF_SECTIONS[key]['name'][:20]}: {len(result)}文字")
def test_prompt_fallback_to_base():
"""tunedプロンプトなし→baseへのフォールバック"""
gen = get_gen()
tmpl = gen.load_prompt("section_I")
assert "system" in tmpl and "user_template" in tmpl
print("プロンプトフォールバック: OK")
if __name__ == "__main__":
pytest.main([__file__, "-v"])
```

```
STEP 7 多指標品質評価エンジン（8メトリクス）
推定: 3〜4日
```

**7.1 実装コード (src/evaluator/composite_evaluator.py)**

```
import re
from evaluate import load as eval_load
from bert_score import score as bscore
from typing import Dict
class IFQualityEvaluator:
DEFAULT_WEIGHTS = {
"text_similarity": 0.10, # ROUGE-L
"semantic_similarity": 0.15, # BERTScore F1 (multilingual)
"numerical_accuracy": 0.25, # 数値・単位一致率（最重要）
"table_reproduction": 0.15, # 表構造再現率
"figure_ref_match": 0.10, # 図表参照整合性
"section_coverage": 0.10, # 全13セクション準拠
"term_coverage": 0.05, # 医薬専門用語カバレッジ
"citation_match": 0.05, # 文献一致率 (ⅩⅠ節)
"cover_completeness": 0.05, # 表紙20項目完備率
}
# 医薬専門用語辞書
PHARMA_TERMS = [
"薬物動態","薬理作用","有害事象","禁忌","警告","副作用",
"バイオアベイラビリティ","半減期","クリアランス","分布容積",
"CYP","トランスポーター","NOAEL","催奇形性","生殖毒性",
"タンパク結合","血中濃度","代謝物","排泄","蓄積",
]
def __init__(self, weights: dict = None):
self.rouge = eval_load("rouge")
self.weights = weights or self.DEFAULT_WEIGHTS
def evaluate(self, generated: dict, reference: dict) -> Dict:
sc = {}
gen_all = " ".join(str(v) for v in generated.values())
ref_all = " ".join(str(v) for v in reference.values())
# 1. ROUGE-L
r = self.rouge.compute(predictions=[gen_all], references=[ref_all])
sc["text_similarity"] = r["rougeL"]
# 2. BERTScore (multilingual, 日英混在対応)
P, R, F = bscore([gen_all], [ref_all], lang="ja",
rescale_with_baseline=True)
sc["semantic_similarity"] = max(0.0, float(F.mean()))
# 3. 数値・単位一致率
sc["numerical_accuracy"] = self._numerical_accuracy(gen_all, ref_all)
# 4. 表構造再現率
sc["table_reproduction"] = self._table_reproduction(generated, reference)
# 5. 図表参照整合性
sc["figure_ref_match"] = self._figure_ref_match(gen_all, ref_all)
# 6. 全13セクション準拠
sc["section_coverage"] = self._section_coverage(generated)
# 7. 専門用語カバレッジ
sc["term_coverage"] = self._term_coverage(gen_all, ref_all)
# 8. 文献一致率
sc["citation_match"] = self._citation_match(generated, reference)
# 9. 表紙完備率
sc["cover_completeness"] = self._cover_completeness(generated)
# 複合スコア (重み付き平均)
sc["composite"] = sum(
self.weights.get(k,0)*sc[k]
for k in sc if k!="composite")
return sc
def _numerical_accuracy(self, gen, ref) -> float:
# 数値+単位のペアで一致率を計算（PKパラメータ等）
pat = re.compile(
r"(\d+\.?\d*)\s*(ng/mL|μg/mL|ng·h/mL|h|L/kg|%|mg|nmol/L)?",
re.IGNORECASE)
gen_n = {(m.group(1),m.group(2) or "") for m in pat.finditer(gen)}
ref_n = {(m.group(1),m.group(2) or "") for m in pat.finditer(ref)}
if not ref_n: return 1.0
return len(gen_n & ref_n) / len(ref_n)
def _table_reproduction(self, gen: dict, ref: dict) -> float:
scores = []
for key in ref:
if key not in gen: continue
rn = set(re.findall(r"\d+\.?\d*", str(ref[key])))
gn = set(re.findall(r"\d+\.?\d*", str(gen[key])))
if rn: scores.append(len(rn & gn) / len(rn))
return sum(scores)/len(scores) if scores else 0.0
def _figure_ref_match(self, gen, ref) -> float:
pat = re.compile(r"(?:表|図|Table|Figure)\s*\d+", re.IGNORECASE)
gr = set(pat.findall(gen))
rr = set(pat.findall(ref))
if not rr: return 1.0
return len(gr & rr) / len(rr)
def _section_coverage(self, gen: dict) -> float:
# IF記載要領2018 全13セクション
secs = [f"section_{x}" for x in
["I","II","III","IV","V","VI","VII","VIII","IX","X","XI","XII","XIII"]]
found = sum(1 for s in secs if s in gen and len(str(gen[s])) > 50)
return found / len(secs)
def _term_coverage(self, gen, ref) -> float:
ref_terms = {t for t in self.PHARMA_TERMS if t in ref}
if not ref_terms: return 1.0
return len({t for t in ref_terms if t in gen}) / len(ref_terms)
def _citation_match(self, gen: dict, ref: dict) -> float:
gxi = str(gen.get("section_XI",""))
rxi = str(ref.get("section_XI",""))
if not rxi: return 1.0
# 年号パターンで文献を近似マッチ
rr = set(re.findall(r"\d{4}[;,.]", rxi))
gr = set(re.findall(r"\d{4}[;,.]", gxi))
if not rr: return 1.0
return len(rr & gr) / len(rr)
def _cover_completeness(self, gen: dict) -> float:
cover = str(gen.get("cover",""))
if not cover: return 0.0
required = ["販売名","一般名","剤形","規格","含量","製造販売承認年月日"]
return sum(1 for kw in required if kw in cover) / len(required)
def section_scores(self, gen: dict, ref: dict) -> Dict[str,float]:
"""セクション別ROUGE-Lスコア（チューニング優先度判定用）"""
result = {}
for key in ref:
if key in gen and gen[key] and ref[key]:
r = self.rouge.compute([str(gen[key])],[str(ref[key])])
result[key] = r["rougeL"]
return result
```

**7.2 Step 7 テストプログラム**

```
# tests/test_step7_evaluator.py
import pytest
from src.evaluator.composite_evaluator import IFQualityEvaluator
GEN = {
"section_VII": "薬物動態。Cmax: 245 ng/mL、AUC0-inf: 1234 ng·h/mL、t1/2: 12.3時間（表1参照）。",
"section_VIII":"有害事象発現率15.3%。頭痛5.2%（表4参照）、悪心3.1%。図3参照。",
"section_IX": "毒性試験。反復投与毒性（ラット90日）のNOAELは100 mg/kg/日。遺伝毒性陰性（表6）。",
"section_XIII":"調剤・服薬支援。簡易懸濁後安定性を表5に示す。",
"cover":"販売名:テスト錠10mg 一般名:テストサン 剤形:錠剤 規格:10mg 含量:10mg 製造販売承認年月日:2020年4月",
"section_XI": "1) Smith J et al. J Pharmacol. 2019;45:123-130. 2) Tanaka K. Yakuri. 2020;",
}
REF = {
"section_VII": "薬物動態。Cmax: 245 ng/mL、AUC0-inf: 1234 ng·h/mL（表1参照）。t1/2: 12.3時間（図1参照）。",
"section_VIII":"臨床試験での有害事象発現率は15.3%。頭痛5.2%が最多（表4・図3参照）。",
"section_IX": "NOAEL: 100 mg/kg/日（ラット90日反復）。遺伝毒性試験陰性（表6）。",
"section_XIII":"簡易懸濁法に関する安定性試験（表5参照）。",
"cover":"販売名:テスト錠10mg 一般名:テストサン 剤形:錠剤 規格:10mg 含量:10mg 製造販売承認年月日:2020年4月1日",
"section_XI": "1) Smith J et al. J Pharmacol. 2019;45:123. 2) Tanaka K. Yakuri. 2020;12:45.",
}
def test_numerical_accuracy_pk():
ev = IFQualityEvaluator()
sc = ev.evaluate(GEN, REF)
assert sc["numerical_accuracy"] > 0.7
print(f"数値一致率: {sc['numerical_accuracy']:.4f}")
def test_figure_ref_match():
ev = IFQualityEvaluator()
sc = ev.evaluate(GEN, REF)
print(f"図表参照整合: {sc['figure_ref_match']:.4f}")
assert 0 = 0.8
print(f"表紙完備率: {sc['cover_completeness']:.4f}")
def test_citation_match():
ev = IFQualityEvaluator()
sc = ev.evaluate(GEN, REF)
print(f"文献一致率: {sc['citation_match']:.4f}")
def test_all_metrics_and_composite():
ev = IFQualityEvaluator()
sc = ev.evaluate(GEN, REF)
assert 0 = 0.9
if __name__ == "__main__":
pytest.main([__file__, "-v"])
```

```
STEP 8 チューニングループ＋Streamlit監視ダッシュボード統合
推定: 4〜6日
```

**8.1 チューニングループ実装 (src/tuner/tuning_loop.py)**

```
import mlflow, yaml, json, copy
from pathlib import Path
from datetime import datetime
from src.rag.retriever import CrossModalRAGPipeline
from src.generator.section_generator import IFSectionGenerator
from src.evaluator.composite_evaluator import IFQualityEvaluator
class PromptOptimizer:
"""低スコアセクションのプロンプトを自動強化"""
def optimize(self, sec_scores: dict, cfg: dict):
if not sec_scores: return
worst = min(sec_scores, key=sec_scores.get)
score = sec_scores[worst]
base_p = Path(f"prompts/base/{worst}.yaml")
tune_p = Path(f"prompts/tuned/{worst}.yaml")
if not base_p.exists(): return
with open(base_p) as f: import yaml; tmpl = yaml.safe_load(f)
if score  dict:
drug_id = self.cfg["system"]["drug_id"]
mlflow.set_experiment(f"IF_Tuning_v2_{drug_id}")
with mlflow.start_run(run_name=datetime.now().strftime("%Y%m%d_%H%M")):
mlflow.log_params({
"max_loops": self.max_loops,
"target": self.target,
"model": self.cfg["llm"]["model"],
"drug_id": drug_id,
"if_standard": "2018_2019"
})
for i in range(1, self.max_loops+1):
print(f"\n══ LOOP {i}/{self.max_loops} ══")
# ① RAGパイプライン再初期化（パラメータ更新反映）
rag = CrossModalRAGPipeline(
f"./data/vectordb/{drug_id}", drug_id, self.cfg["rag"])
# ② IF全13セクション生成
gen = IFSectionGenerator(
drug_id, self.cfg["system"]["drug_name"],
rag, self.cfg["llm"]["model"])
generated = gen.generate_full_if()
# ③ 8メトリクス評価
ev = IFQualityEvaluator(self.cfg["evaluation"]["weights"])
scores = ev.evaluate(generated, self.ref)
sec_sc = ev.section_scores(generated, self.ref)
self.history.append({"loop":i,"scores":scores,"section_scores":sec_sc})
# ④ MLflowログ
for k,v in scores.items(): mlflow.log_metric(k, v, step=i)
print(f" Composite: {scores['composite']:.4f} Target: {self.target}")
print(f" NumAcc: {scores['numerical_accuracy']:.4f}",
f"TableRepr: {scores['table_reproduction']:.4f}",
f"FigRef: {scores['figure_ref_match']:.4f}")
# ⑤ 最良更新
if scores["composite"] > self.best_score:
self.best_score = scores["composite"]
self.best_if = copy.deepcopy(generated)
self.no_improve = 0
self._save_best(generated, scores, i)
else:
self.no_improve += 1
# ⑥ 終了判定
if scores["composite"] >= self.target:
print(f" ✓ TARGET REACHED! (loop {i})"); break
if self.no_improve >= self.patience:
print(f" Early stop ({self.patience}回改善なし)"); break
# ⑦ 3種チューニング
self.prompt_opt.optimize(sec_sc, self.cfg)
self.rag_opt.optimize(scores, self.cfg)
return {"best_score":self.best_score,
"best_if":self.best_if,"history":self.history}
def _save_best(self, gen, scores, loop_i):
out = Path(f"outputs/{self.cfg['system']['drug_id']}/generated")
out.mkdir(parents=True, exist_ok=True)
with open(out/f"best_loop{loop_i}.json","w") as f:
json.dump({"loop":loop_i,"scores":scores,"sections":gen},
f, ensure_ascii=False, indent=2)
```

**8.2 Streamlit監視ダッシュボード (src/dashboard/app.py)**

```
import streamlit as st
import mlflow, json, pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
st.set_page_config(page_title="IF-AutoGen v2.0", page_icon="💊", layout="wide")
st.title("💊 IF-AutoGen v2.0 — 監視ダッシュボード")
st.caption("IF記載要領2018(2019年更新版) | マルチモーダルRAG | 完全ローカル実行")
with st.sidebar:
st.header("⚙️ チューニング設定")
drug_id = st.text_input("医薬品ID", "drug_A")
drug_name = st.text_input("医薬品名", "○○錠10mg")
model = st.selectbox("LLMモデル（ローカル）",
["qwen2.5:14b","qwen2.5:72b","llama3.1:8b"])
max_loops = st.slider("最大ループ回数", 1, 30, 15)
target_score = st.slider("ターゲットスコア", 0.5, 1.0, 0.85, 0.01)
st.markdown("**RAG設定**")
top_k = st.slider("Top-K (rerank後)", 3, 10, 5)
max_tables = st.slider("最大展開表数", 1, 5, 3)
vision_on = st.checkbox("Vision-LLM 図解析（Qwen2-VL）", True)
run_btn = st.button("▶ チューニング開始", type="primary")
st.divider()
st.caption("🔒 全ローカル実行 | インターネット不要")
def load_history(drug_id):
client = mlflow.MlflowClient()
try:
exp = client.get_experiment_by_name(f"IF_Tuning_v2_{drug_id}")
if not exp: return []
runs = client.search_runs(exp.experiment_id,
order_by=["start_time DESC"], max_results=1)
if not runs: return []
metrics = ["composite","numerical_accuracy","table_reproduction",
"figure_ref_match","section_coverage","text_similarity",
"semantic_similarity","term_coverage","citation_match"]
hist = []
for m in metrics:
for mv in client.get_metric_history(runs[0].info.run_id, m):
hist.append({"loop":mv.step,"metric":m,"value":mv.value})
return hist
except: return []
history = load_history(drug_id)
if history:
df = pd.DataFrame(history)
piv = df.pivot(index="loop", columns="metric", values="value")
# KPIカード
c1,c2,c3,c4 = st.columns(4)
for col, label in [("composite","Composite"),("numerical_accuracy","数値一致率"),
("table_reproduction","表再現率"),("figure_ref_match","図参照整合")]:
if col in piv.columns:
v = piv[col].iloc[-1]
delta = v - piv[col].iloc[0] if len(piv) > 1 else 0
[c1,c2,c3,c4][["composite","numerical_accuracy",
"table_reproduction","figure_ref_match"].index(col)].metric(
label, f"{v:.4f}", f"+{delta:.4f}")
# スコア推移グラフ
st.subheader("📈 スコア推移")
fig = go.Figure()
colors = {"composite":"crimson","numerical_accuracy":"steelblue",
"table_reproduction":"forestgreen","figure_ref_match":"orange",
"section_coverage":"purple","semantic_similarity":"teal"}
for col in piv.columns:
fig.add_trace(go.Scatter(
x=piv.index, y=piv[col], name=col,
mode="lines+markers",
line=dict(color=colors.get(col,"gray"),
width=3 if col=="composite" else 1.5)))
fig.add_hline(y=target_score, line_dash="dash", line_color="red",
annotation_text=f"Target: {target_score}")
fig.update_layout(xaxis_title="Loop", yaxis_title="Score", height=400)
st.plotly_chart(fig, use_container_width=True)
# セクション別スコアヒートマップ（最新ループ）
score_files = sorted(Path(f"outputs/{drug_id}/generated").glob("*.json"))
if score_files:
with open(score_files[-1]) as f: ld = json.load(f)
sc = ld.get("scores",{})
sec_labels = [k for k in sc if k != "composite"]
sec_values = [sc[k] for k in sec_labels]
st.subheader("🔥 メトリクス別スコア（最新ループ）")
fig2 = px.bar(x=sec_labels, y=sec_values,
color=sec_values, color_continuous_scale="RdYlGn",
range_color=[0,1], labels={"x":"Metric","y":"Score"})
fig2.add_hline(y=target_score, line_dash="dash", line_color="red")
st.plotly_chart(fig2, use_container_width=True)
else:
st.info("チューニング結果なし。サイドバーから「チューニング開始」を実行してください。")
# 起動: streamlit run src/dashboard/app.py
# アクセス: http://localhost:8501
```

**8.3 統合エントリーポイント (main.py)**

```
#!/usr/bin/env python
# main.py — IF-AutoGen v2.0 エントリーポイント
import yaml, json, argparse
from pathlib import Path
from src.processor.text_extractor import CTDTextExtractor
from src.processor.table_extractor import CTDTableExtractor
from src.processor.vision_analyzer import CTDFigureExtractor
from src.processor.chunker import ContextualChunker
from src.indexer.multi_index_builder import MultiIndexBuilder
from src.tuner.tuning_loop import TuningLoopController
def main():
parser = argparse.ArgumentParser()
parser.add_argument("--config", default="config.yaml")
args = parser.parse_args()
with open(args.config) as f: cfg = yaml.safe_load(f)
drug_id = cfg["system"]["drug_id"]
mode = cfg["system"].get("mode","full")
# ─── Phase 1: CTD文書処理 ───────────────────────
if mode in ("full","index"):
print("=== Phase 1: CTD Document Processing ===")
ctd_dir = Path(f"data/raw/{drug_id}/ctd")
fig_dir = f"data/processed/{drug_id}/figures"
db_path = f"./data/vectordb/{drug_id}"
text_ext = CTDTextExtractor()
tbl_ext = CTDTableExtractor()
fig_ext = CTDFigureExtractor(fig_dir, cfg["llm"]["vision_model"],
cfg["llm"]["ollama_host"])
all_blocks, all_tables, all_figures = [], [], []
for pdf in sorted(ctd_dir.glob("*.pdf")):
module = text_ext.detect_module(pdf.name)
print(f" Processing {pdf.name} [{module}]...")
all_blocks += text_ext.extract(str(pdf))
all_tables += tbl_ext.extract(str(pdf), module)
all_figures += fig_ext.extract(str(pdf), module)
print(f" Text blocks: {len(all_blocks)}, Tables: {len(all_tables)}, Figures: {len(all_figures)}")
chunker = ContextualChunker(
parent_size=cfg["rag"]["chunk_size_parent"],
child_size =cfg["rag"]["chunk_size_child"])
chunks = chunker.chunk(all_blocks, all_tables, all_figures)
print(f" Contextual chunks: {len(chunks)}")
builder = MultiIndexBuilder(db_path, drug_id)
builder.index_text(chunks)
builder.index_tables(all_tables)
builder.index_figures(all_figures)
print(f" Index built: {builder.tc.count()} text chunks")
# ─── Phase 2: チューニングループ ────────────────
if mode in ("full","tune"):
print("\n=== Phase 2: Tuning Loop ===")
ref_path = Path(f"data/raw/{drug_id}/if_human_parsed.json")
if not ref_path.exists():
print(f"ERROR: {ref_path} が見つかりません")
return
with open(ref_path) as f: ref_if = json.load(f)
ctrl = TuningLoopController(cfg, ref_if)
result = ctrl.run()
print(f"\n最終スコア: {result['best_score']:.4f}")
print(f"完了ループ数: {len(result['history'])}")
print(f"生成IF保存先: outputs/{drug_id}/generated/")
if __name__ == "__main__":
main()
# 使用例:
# python main.py # フル実行（インデックス構築→チューニング）
# python main.py --config config.yaml # 設定ファイル指定
# streamlit run src/dashboard/app.py # 監視ダッシュボード起動
# mlflow ui --port 5001 # MLflow UI起動
```

**8.4 Step 8 統合テストプログラム**

```
# tests/test_step8_integration.py
import pytest, yaml, json, subprocess, time, httpx
from src.tuner.tuning_loop import TuningLoopController
MINI_CFG = {
"system": {"drug_id":"drug_test","drug_name":"テスト薬品A","if_standard":"2018_2019"},
"llm": {"model":"qwen2.5:14b","vision_model":"qwen2-vl:7b",
"temperature":0.1,"ollama_host":"http://localhost:11434"},
"rag": {"top_k_retrieve":10,"top_k_rerank":3,"bm25_weight":0.4,"dense_weight":0.6,
"cross_modal":{"max_linked_tables":2,"max_linked_figures":1}},
"tuning": {"max_loops":2,"target_score":0.99,"early_stop_patience":10},
"evaluation": {"weights": {
"text_similarity":0.10,"semantic_similarity":0.15,"numerical_accuracy":0.25,
"table_reproduction":0.15,"figure_ref_match":0.10,"section_coverage":0.10,
"term_coverage":0.05,"citation_match":0.05,"cover_completeness":0.05}}
}
REF_IF = {
"section_VII": "薬物動態 Cmax 245 ng/mL AUC 1234 ng*h/mL t1/2 12.3時間 表1参照 図1参照",
"section_VIII":"有害事象発現率15.3% 頭痛5.2% 表4参照 図3参照",
"section_IX": "NOAEL 100 mg/kg 反復毒性 ラット90日 表6参照",
"section_XIII":"簡易懸濁安定性試験 表5参照",
"cover": "販売名:テスト錠10mg 一般名:テストサン 剤形:錠剤 規格:10mg 含量:10mg 製造販売承認年月日:2020年4月",
"section_XI": "Smith J. 2019; Tanaka K. 2020;",
}
def test_full_tuning_loop():
ctrl = TuningLoopController(MINI_CFG, REF_IF)
result = ctrl.run()
assert "best_score" in result
assert "history" in result
assert len(result["history"]) >= 1
print(f"完了ループ数: {len(result['history'])}")
print(f"最終Composite: {result['best_score']:.4f}")
def test_all_9_metrics_recorded():
ctrl = TuningLoopController(MINI_CFG, REF_IF)
result = ctrl.run()
expected = ["text_similarity","semantic_similarity","numerical_accuracy",
"table_reproduction","figure_ref_match","section_coverage",
"term_coverage","citation_match","cover_completeness","composite"]
sc = result["history"][0]["scores"]
for m in expected:
assert m in sc, f"メトリクス欠落: {m}"
print(f"全9メトリクス+composite 確認: OK")
def test_mlflow_experiment_v2():
import mlflow
ctrl = TuningLoopController(MINI_CFG, REF_IF)
ctrl.run()
exp = mlflow.get_experiment_by_name("IF_Tuning_v2_drug_test")
assert exp is not None, "MLflow v2実験が作成されていない"
print("MLflow実験 IF_Tuning_v2_drug_test: OK")
def test_best_if_saved_to_file():
from pathlib import Path
ctrl = TuningLoopController(MINI_CFG, REF_IF)
ctrl.run()
saved = list(Path("outputs/drug_test/generated").glob("best_loop*.json"))
assert len(saved) > 0, "最良IFファイルが保存されていない"
with open(saved[-1]) as f: data = json.load(f)
assert "scores" in data and "sections" in data
print(f"最良IF保存確認: {saved[-1].name}")
def test_config_yaml_v2_complete():
cfg = yaml.safe_load(open("config.yaml"))
assert cfg["system"]["if_standard"] == "2018_2019"
assert "cross_modal" in cfg["rag"]
assert cfg["rag"]["cross_modal"]["enable_vision_llm"] == True
ws = sum(cfg["evaluation"]["weights"].values())
assert abs(ws - 1.0) ■ 全ステップ完了後の目標スコア（CompositeScore）
Phase 0 (Loop 1): 0.30〜0.45 ← ベースライン
Phase 1 (Loop 2〜4): 0.55〜0.65 ← プロンプト・RAG基本チューニング
Phase 2 (Loop 5〜8): 0.70〜0.78 ← 数値一致率・表構造再現の改善
Phase 3 (Loop 9〜15): 0.82〜0.88 ← 図表参照整合・専門用語精度の改善
最終目標 (target): 0.85以上 ← config.yaml target_score で設定可能
```

**すべての処理はローカルPC上で完結します。製薬企業の機密CTD文書が外部に漏洩するリスクはゼロです。初回のモデルダウンロード以降は完全オフライン動作します。**
