import mlflow, json, yaml, copy
from pathlib import Path
from datetime import datetime
from src.rag.retriever import CrossModalRAGPipeline
from src.generator.section_generator import IFSectionGenerator
from src.evaluator.composite_evaluator import IFQualityEvaluator

class PromptOptimizer:
    def optimize(self, sec_scores: dict, cfg: dict):
        if not sec_scores: return
        worst = min(sec_scores, key=sec_scores.get)
        score = sec_scores[worst]
        base_p = Path(f"prompts/base/{worst}.yaml")
        tune_p = Path(f"prompts/tuned/{worst}.yaml")
        if not base_p.exists(): return
        
        with open(base_p, encoding="utf-8") as f: tmpl = yaml.safe_load(f)
        if score < 0.6:
            # Simple prompt boost for low score section
            tmpl["system"] += "\n【重要】このセクションはより詳細にかつ、正確な数値を漏れなく記述してください。"
            tune_p.parent.mkdir(parents=True, exist_ok=True)
            with open(tune_p, "w", encoding="utf-8") as f: yaml.dump(tmpl, f, allow_unicode=True)

class TuningLoopController:
    def __init__(self, cfg, ref_if):
        self.cfg = cfg
        self.ref = ref_if
        self.max_loops = cfg["tuning"]["max_loops"]
        self.target = cfg["tuning"]["target_score"]
        self.patience = cfg["tuning"]["early_stop_patience"]
        self.history = []
        self.best_score = 0.0
        self.best_if = None
        self.no_improve = 0
        self.prompt_opt = PromptOptimizer()

    def run(self):
        drug_id = self.cfg["system"]["drug_id"]
        mlflow.set_experiment(f"IF_Tuning_v2_{drug_id}")
        
        with mlflow.start_run(run_name=datetime.now().strftime("%Y%m%d_%H%M")):
            mlflow.log_params({"max_loops": self.max_loops, "target": self.target, "model": self.cfg["llm"]["model"]})
            
            for i in range(1, self.max_loops + 1):
                print(f"\n--- LOOP {i}/{self.max_loops} ---")
                
                # 1. Pipeline Re-init
                rag = CrossModalRAGPipeline(f"./data/vectordb/{drug_id}", drug_id, self.cfg["rag"])
                gen = IFSectionGenerator(drug_id, self.cfg["system"]["drug_name"], rag, self.cfg["llm"]["model"])
                
                # 2. Generate
                generated = gen.generate_full_if()
                
                # 3. Evaluate
                ev = IFQualityEvaluator(self.cfg["evaluation"]["weights"])
                scores = ev.evaluate(generated, self.ref)
                sec_sc = ev.section_scores(generated, self.ref)
                self.history.append({"loop": i, "scores": scores, "section_scores": sec_sc})
                
                # 4. Logs
                for k, v in scores.items(): mlflow.log_metric(k, v, step=i)
                print(f"  Composite: {scores['composite']:.4f} / Target: {self.target}")
                
                # 5. Best tracking & Save
                if scores["composite"] > self.best_score:
                    self.best_score = scores["composite"]
                    self.best_if = copy.deepcopy(generated)
                    self.no_improve = 0
                    self._save_best(generated, scores, i)
                else:
                    self.no_improve += 1
                
                # 6. Early Stopping / Target
                if scores["composite"] >= self.target:
                    print("  Target reached!"); break
                if self.no_improve >= self.patience:
                    print("  Early stopping."); break
                
                # 7. Prompt Optimization
                if self.cfg["tuning"].get("enable_prompt_tuning"):
                    self.prompt_opt.optimize(sec_sc, self.cfg)
                    
        return {"best_score": self.best_score, "best_if": self.best_if, "history": self.history}

    def _save_best(self, gen, scores, loop_i):
        out = Path(f"outputs/{self.cfg['system']['drug_id']}/generated")
        out.mkdir(parents=True, exist_ok=True)
        with open(out / f"best_loop{loop_i}.json", "w", encoding="utf-8") as f:
            json.dump({"loop": loop_i, "scores": scores, "sections": gen}, f, ensure_ascii=False, indent=2)
