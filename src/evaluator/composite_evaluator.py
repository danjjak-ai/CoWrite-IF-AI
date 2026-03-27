import re
from evaluate import load as eval_load
from bert_score import score as bscore
from typing import Dict

class IFQualityEvaluator:
    DEFAULT_WEIGHTS = {
        "text_similarity": 0.10,     # ROUGE-L
        "semantic_similarity": 0.15, # BERTScore F1
        "numerical_accuracy": 0.25,  # Match of numbers + units
        "table_reproduction": 0.15,  # Table coverage
        "figure_ref_match": 0.10,    # Figure/Table referencing consistency
        "section_coverage": 0.10,    # All 13 sections presence
        "term_coverage": 0.05,       # Pharma terms
        "citation_match": 0.05,      # References
        "cover_completeness": 0.05,  # Cover items
    }
    
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
        
        # 2. BERTScore (Japanese/Multilingual)
        P, R, F = bscore([gen_all], [ref_all], lang="ja")
        sc["semantic_similarity"] = float(F.mean())
        
        # 3. Numerical Accuracy
        sc["numerical_accuracy"] = self._numerical_accuracy(gen_all, ref_all)
        
        # 4. Table Reproduction (approximate by matching numbers in table sections)
        sc["table_reproduction"] = self._table_reproduction(generated, reference)
        
        # 5. Figure/Table Reference Match
        sc["figure_ref_match"] = self._figure_ref_match(gen_all, ref_all)
        
        # 6. Section Coverage
        sc["section_coverage"] = self._section_coverage(generated)
        
        # 7. Term Coverage
        sc["term_coverage"] = self._term_coverage(gen_all, ref_all)
        
        # 8. Citation & Cover (omitted for speed or simplified)
        sc["citation_match"] = 1.0 # placeholder
        sc["cover_completeness"] = 1.0 # placeholder
        
        # Composite Score
        sc["composite"] = sum(self.weights.get(k,0)*sc[k] for k in sc if k!="composite")
        return sc

    def _numerical_accuracy(self, gen, ref) -> float:
        pat = re.compile(r"(\d+\.?\d*)\s*(ng/mL|μg/mL|h|%|mg|nmol/L)?", re.IGNORECASE)
        gen_n = set(pat.findall(gen))
        ref_n = set(pat.findall(ref))
        if not ref_n: return 1.0
        return len(gen_n & ref_n) / len(ref_n)

    def _table_reproduction(self, gen, ref) -> float:
        # Simple number-based set match per section
        scores = []
        for k in ref:
            if k in gen:
                rn = set(re.findall(r"\d+\.?\d*", str(ref[k])))
                gn = set(re.findall(r"\d+\.?\d*", str(gen[k])))
                if rn: scores.append(len(rn & gn) / len(rn))
        return sum(scores)/len(scores) if scores else 0.0

    def _figure_ref_match(self, gen, ref) -> float:
        pat = re.compile(r"(?:表|図|Table|Figure)\s*\d+", re.IGNORECASE)
        gr = set(pat.findall(gen))
        rr = set(pat.findall(ref))
        if not rr: return 1.0
        return len(gr & rr) / len(rr)

    def _section_coverage(self, gen: dict) -> float:
        from src.generator.section_generator import IF_SECTIONS
        found = sum(1 for s in IF_SECTIONS if s in gen and len(str(gen[s])) > 100)
        return found / len(IF_SECTIONS)

    def _term_coverage(self, gen, ref) -> float:
        ref_terms = {t for t in self.PHARMA_TERMS if t in ref}
        if not ref_terms: return 1.0
        return len({t for t in ref_terms if t in gen}) / len(ref_terms)

    def section_scores(self, gen, ref) -> dict:
        result = {}
        for k in ref:
            if k in gen and gen[k] and ref[k]:
                r = self.rouge.compute(predictions=[str(gen[k])], references=[str(ref[k])])
                result[k] = r["rougeL"]
        return result
