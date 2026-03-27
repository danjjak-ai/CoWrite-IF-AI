import streamlit as st
import mlflow, json, pandas as pd, yaml
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import os

st.set_page_config(page_title="IF-AutoGen v2.0", page_icon="💊", layout="wide")
st.title("💊 IF-AutoGen v2.0 — 監視ダッシュボード")

# Helper to scan processed drugs
@st.cache_data
def get_processed_drugs():
    drugs = []
    # Use absolute path to ensure robustness
    output_base = Path(os.getcwd()) / "outputs"
    if not output_base.exists(): return []
    
    for drug_path in output_base.iterdir():
        if drug_path.is_dir():
            drug_id = drug_path.name
            # Check loops
            gen_path = drug_path / "generated"
            loops = 0
            latest_score = 0.0
            if gen_path.exists():
                loop_files = list(gen_path.glob("best_loop*.json"))
                if loop_files:
                    # Get the numbers and find max
                    nums = [int(f.stem.replace("best_loop", "")) for f in loop_files]
                    loops = max(nums)
                    # Peek at score of the latest loop
                    try:
                        latest_file = gen_path / f"best_loop{loops}.json"
                        with open(latest_file, encoding="utf-8") as f:
                            data = json.load(f)
                            latest_score = data.get("scores", {}).get("composite", 0.0)
                    except: pass
            
            drugs.append({"id": drug_id, "loops": loops, "score": latest_score})
    return drugs

with st.sidebar:
    st.header("⚙️ 設定")
    processed_drugs = get_processed_drugs()
    
    if processed_drugs:
        drug_labels = [f"💊 {d['id']} (Loops: {d['loops']})" for d in processed_drugs]
        selected_label = st.selectbox("処理済み医薬品を選択", drug_labels)
        # Extract ID from label
        selected_id = selected_label.split(" ")[1]
        
        # Check if we should override ID or show static
        st.info(f"선택됨: {selected_id}")
    else:
        st.warning("処理済みデータが見つかりません。")
        selected_id = st.text_input("医薬品IDを直接入力", "drug_A")
    
    st.divider()
    st.caption("🔒 全ローカル実行 | インターネット不要")

# Load history
def load_history(drug_id):
    try:
        client = mlflow.MlflowClient()
        exp = client.get_experiment_by_name(f"IF_Tuning_v2_{drug_id}")
        if not exp: return []
        runs = client.search_runs(exp.experiment_id, order_by=["start_time DESC"], max_results=1)
        if not runs: return []
        
        metrics = ["composite","numerical_accuracy","table_reproduction","figure_ref_match","section_coverage"]
        hist = []
        for m in metrics:
            try:
                for mv in client.get_metric_history(runs[0].info.run_id, m):
                    hist.append({"loop": mv.step, "metric": m, "value": mv.value})
            except: pass
        return hist
    except: return []

hist = load_history(selected_id)
if hist:
    df = pd.DataFrame(hist)
    st.subheader("📈 スコア推移")
    fig = px.line(df, x="loop", y="value", color="metric", title=f"{selected_id} 성능 추이", markers=True)
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info(f"{selected_id} 에 대한 MLflow 기록이 없거나 아직 진행 중입니다.")

st.divider()
st.subheader("📄 生成結果 vs 使用プロンプト")

output_path = Path(os.getcwd()) / "outputs" / selected_id / "generated"
if output_path.exists():
    best_files = sorted(output_path.glob("best_loop*.json"), reverse=True)
    if best_files:
        # Load the BEST (latest) file
        with open(best_files[0], encoding="utf-8") as f: data = json.load(f)
        st.success(f"최종 결과 표시 중: Loop {data.get('loop')} (종합 점수: {data.get('scores',{}).get('composite',0):.4f})")
        
        sections = data.get("sections", {})
        for sec_id, content in sections.items():
            with st.expander(f"📌 {sec_id}"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**[Generated Content]**")
                    st.write(content)
                
                with col2:
                    st.markdown("**[Final Prompt Content]**")
                    # Try to load prompt (tuned priority)
                    # Check in tuned first
                    prompt_file = Path(f"prompts/tuned/{sec_id}.yaml")
                    if not prompt_file.exists():
                        # Fallback to base
                        prompt_file = Path(f"prompts/base/{sec_id}.yaml")
                    
                    if prompt_file.exists():
                        try:
                            with open(prompt_file, encoding="utf-8") as pf:
                                p_data = yaml.safe_load(pf)
                                st.code(p_data.get("system", ""), language="yaml")
                        except Exception as e:
                            st.error(f"Error loading prompt: {e}")
                    else:
                        st.info("기본/최종 프롬프트 파일을 찾을 수 없습니다.")
    else:
        st.info("생성된 섹션 파일이 없습니다.")
else:
    st.info(f"{selected_id} 에 대한 생성 결과 폴더가 없습니다.")
