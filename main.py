#!/usr/bin/env python
# main.py — IF-AutoGen v2.0 エントリーポイント
import yaml, json, argparse, os
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
    parser.add_argument("--mode", default="full", help="index | generate | tune | full")
    parser.add_argument("--drug_id", help="Drug ID to process (overrides config)")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Use CLI arg if provided, otherwise config
    drug_id = args.drug_id or cfg["system"]["drug_id"]
    cfg["system"]["drug_id"] = drug_id

    db_path = f"data/vectordb/{drug_id}"
    out_dir = Path(f"data/processed/{drug_id}")

    # PHASE 1: Indexing
    if args.mode in ("full", "index") and not os.path.exists(db_path):
        print("=== Phase 1: CTD Document Processing & Indexing ===")
        # (This part is already in run_pipeline_step2_5.py)
        import subprocess
        subprocess.run([".venv/Scripts/python", "run_pipeline_step2_5.py"])

    # PHASE 2: Generation / Tuning
    if args.mode in ("full", "generate", "tune"):
        print("\n=== Phase 2: Generation / Tuning Loop ===")
        ref_path = Path(f"data/raw/{drug_id}/if_human_parsed.json")
        
        if not ref_path.exists():
            print(f"INFO: Reference {ref_path} not found. Running generation only.")
            # Run one generation cycle
            from src.rag.retriever import CrossModalRAGPipeline
            from src.generator.section_generator import IFSectionGenerator
            
            rag = CrossModalRAGPipeline(db_path, drug_id, cfg["rag"])
            gen = IFSectionGenerator(drug_id, cfg["system"]["drug_name"], rag, cfg["llm"]["model"])
            generated = gen.generate_full_if()
            
            # Save the result
            out = Path(f"outputs/{drug_id}/generated")
            out.mkdir(parents=True, exist_ok=True)
            with open(out / "initial_generation.json", "w", encoding="utf-8") as f:
                json.dump(generated, f, ensure_ascii=False, indent=2)
            print(f"Generation complete. Results in outputs/{drug_id}/generated/initial_generation.json")
        else:
            with open(ref_path, encoding="utf-8") as f:
                ref_if = json.load(f)
            ctrl = TuningLoopController(cfg, ref_if)
            result = ctrl.run()
            print(f"Tuning complete. Best score: {result['best_score']:.4f}")

if __name__ == "__main__":
    main()
