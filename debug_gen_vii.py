import yaml, json, sys, traceback, os
sys.path.append(os.getcwd())

from src.rag.retriever import CrossModalRAGPipeline
from src.generator.section_generator import IFSectionGenerator

def test_gen():
    try:
        with open("config.yaml", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        
        drug_id = cfg["system"]["drug_id"]
        db_path = f"data/vectordb/{drug_id}"
        
        # Override K to 5 for debug speed
        cfg["rag"]["top_k_retrieve"] = 5
        
        print("RAG init...")
        rag = CrossModalRAGPipeline(db_path, drug_id, cfg["rag"])
        
        print("Retrieving VII...")
        ctx = rag.retrieve("薬物動態 パラメータ", if_section="section_VII")
        context_text = ctx.to_llm_input()
        print(f"Context length: {len(context_text)} chars")
        print(f"Tables found: {len(ctx.tables)}, Figures found: {len(ctx.figures)}")
        
        if len(context_text) < 10:
            print("ERROR: Context too short. Is DB empty?")
            return

        print("GEN init...")
        gen = IFSectionGenerator(drug_id, cfg["system"]["drug_name"], rag, cfg["llm"]["model"], host=cfg["llm"]["ollama_host"])
        
        print("GENERATE VII (Fast mode)...")
        # Direct call to see response
        result = gen.generate_section("section_VII")
        
        print(f"RESULT SUCCESS: {len(result)} chars")
        with open("OUT_VII.txt", "w", encoding="utf-8") as f:
            f.write(result)

    except Exception as e:
        traceback.print_exc()

if __name__ == "__main__":
    test_gen()
