# tests/test_step1_env.py
import pytest, httpx, os, sys, yaml

def test_python_version():
    assert sys.version_info >= (3, 11), "Python 3.11+ 必須"

def test_ollama_server_running():
    try:
        r = httpx.get("http://localhost:11434/api/tags", timeout=5)
        assert r.status_code == 200, "Ollamaサーバーが起動していません"
    except Exception as e:
        pytest.fail(f"Ollamaサーバーに接続できません: {e}")

def get_config():
    with open("config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def test_qwen_model_available():
    cfg = get_config()
    target_model = cfg["llm"]["model"]
    r = httpx.get("http://localhost:11434/api/tags")
    models = [m["name"] for m in r.json()["models"]]
    assert any(target_model in m for m in models), f"{target_model} モデル未インストール"

def test_vision_model_available():
    cfg = get_config()
    target_model = cfg["llm"]["vision_model"]
    r = httpx.get("http://localhost:11434/api/tags")
    models = [m["name"] for m in r.json()["models"]]
    assert any(target_model in m for m in models), f"{target_model} Vision-LLM未インストール"

def test_llm_japanese_if_response():
    from ollama import Client
    cfg = get_config()
    model = cfg["llm"]["model"]
    host = cfg["llm"].get("ollama_host", "http://localhost:11434")
    client = Client(host=host)
    try:
        resp = client.chat(model=model,
                          messages=[{"role":"user","content":
                          "医薬品インタビューフォームのⅦ節（薬物動態）では何を記載しますか？一文で答えよ"}])
        text = resp["message"]["content"]
        assert len(text) > 0
        print(f"LLM応答: {text[:100]}")
    except Exception as e:
        pytest.fail(f"LLM推論エラー: {e}")

def test_multilingual_embedding_ja_en():
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer("intfloat/multilingual-e5-large")
    v_ja = m.encode(["passage: 薬物動態 吸収 分布 代謝 排泄 Cmax AUC"])
    v_en = m.encode(["passage: pharmacokinetics absorption distribution metabolism"])
    assert v_ja.shape == (1, 1024), f"次元数異常: {v_ja.shape}"
    from numpy import dot; from numpy.linalg import norm
    sim = dot(v_ja[0],v_en[0])/(norm(v_ja[0])*norm(v_en[0]))
    assert sim > 0.4, f"日英PK用語の意味的類似度が低い: {sim:.3f}"
    print(f"日英PK用語意味的類似度: {sim:.4f}")

def test_sudachipy_medical_terms():
    import sudachipy
    tok = sudachipy.Dictionary().create()
    tokens = [t.surface() for t in tok.tokenize("薬物動態パラメータ解析")]
    assert len(tokens) >= 2
    print(f"SudachiPy分割結果: {tokens}")

def test_chromadb_local_persist():
    import chromadb
    from chromadb.config import Settings
    client = chromadb.Client(Settings(anonymized_telemetry=False))
    col = client.create_collection("test_env")
    col.add(documents=["Cmax 245 ng/mL AUC 1234 ng*h/mL t1/2 12.3 h"],ids=["1"])
    r = col.query(query_texts=["PKパラメータ Cmax"], n_results=1)
    assert "245" in r["documents"][0][0]
    client.delete_collection("test_env")
    print("ChromaDB ローカル動作: OK")

def test_mlflow_local_tracking():
    import mlflow
    with mlflow.start_run(run_name="env_test"):
        mlflow.log_metric("test_score", 1.0)
        mlflow.log_param("model", "qwen2.5-coder:7b")
    print("MLflow ローカルトラッキング: OK")

def test_config_yaml_valid():
    cfg = get_config()
    assert cfg["system"]["if_standard"] == "2018_2019"
    assert cfg["tuning"]["max_loops"] > 0
    assert 0 < cfg["rag"]["bm25_weight"] < 1.0
