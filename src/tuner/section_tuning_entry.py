import sys, yaml, json, argparse, os
from pathlib import Path
sys.path.append(os.getcwd())

from src.rag.retriever import CrossModalRAGPipeline
from src.generator.section_generator import IFSectionGenerator
from src.evaluator.composite_evaluator import IFQualityEvaluator
from parse_reference_if import parse_if_pdf

def run_section_tuning(drug_id, section_id, max_loops):
    # 1. Config loading
    with open("config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    
    # 2. Reference PDF check & Parsing
    ref_pdf = f"data/raw/{drug_id}/if_human.pdf"
    ref_json = f"data/raw/{drug_id}/if_human_parsed.json"
    
    if not os.path.exists(ref_json):
        if os.path.exists(ref_pdf):
            print(f"[*] Reference JSON not found. Parsing {ref_pdf}...")
            parse_if_pdf(ref_pdf, ref_json)
        else:
            print(f"[-] FATAL: {ref_pdf} not found! Please place reference PDF.")
            return

    with open(ref_json, encoding="utf-8") as f:
        ref_data = json.load(f)
    
    # 3. Tuning Loop
    db_path = f"data/vectordb/{drug_id}"
    
    # Check if index exists, if not, try to build it (simplification)
    if not os.path.exists(db_path):
        print(f"[*] VectorDB for {drug_id} not found. please run indexing for this drug first.")
        return

    for i in range(1, max_loops + 1):
        print(f"\n--- [LOOP {i}/{max_loops}] Tuning {section_id} for {drug_id} ---")
        
        # Fresh init for prompt tuning check
        rag = CrossModalRAGPipeline(db_path, drug_id, cfg["rag"])
        gen = IFSectionGenerator(drug_id, cfg["system"]["drug_name"], rag, cfg["llm"]["model"])
        
        # Generation
        print(f"[*] Generating {section_id}...")
        generated_text = gen.generate_section(section_id)
        
        # Evaluation
        ev = IFQualityEvaluator()
        # Single section ROUGE comparison
        if section_id in ref_data:
            score_dict = ev.section_scores({section_id: generated_text}, ref_data)
            score = score_dict.get(section_id, 0.0)
            print(f"[*] Current ROUGE-L Score: {score:.4f}")
        else:
            print(f"[!] Section {section_id} not found in reference data. Scoring skipped.")
            score = 0.0
        
        # Save output
        out_dir = Path(f"outputs/{drug_id}/tuning/{section_id}")
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / f"loop_{i}.txt", "w", encoding="utf-8") as f:
            f.write(generated_text)
        print(f"[*] Result saved to: {out_dir}/loop_{i}.txt")
            
        # Optimization (only if score is low)
        if score < 0.7:
            from src.tuner.tuning_loop import PromptOptimizer
            opt = PromptOptimizer()
            opt.optimize({section_id: score}, cfg)
            print(f"[*] Low score detected. Prompt optimized for next iteration.")
        elif score >= 0.85:
            print(f"[*] High score reached! Ending tuning for this section.")
            break

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--drug", required=True, help="Drug folder name (drug_A, drug_B...)")
    parser.add_argument("--section", required=True, help="Section to tune (section_I to section_XIII)")
    parser.add_argument("--loops", type=int, default=3, help="Max iterations")
    args = parser.parse_args()
    
    run_section_tuning(args.drug, args.section, args.loops)
