from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import chromadb, json, numpy as np, sudachipy
from dataclasses import dataclass, field
from typing import List, Dict

@dataclass
class RetrievalContext:
    text_passages: List[str] = field(default_factory=list)
    tables: Dict[str,str] = field(default_factory=dict)
    figures: Dict[str,str] = field(default_factory=dict)

    def to_llm_input(self) -> str:
        parts = []
        if self.text_passages:
            parts.append("## Reference Text\n" + "\n\n".join(self.text_passages))
        for tid, tdata in self.tables.items():
            parts.append(f"## {tid}\n{tdata}")
        for fid, fdata in self.figures.items():
            parts.append(f"## {fid} (Figure Info)\n{fdata}")
        return "\n\n".join(parts)

class CrossModalRAGPipeline:
    def __init__(self, db_path, drug_id, config):
        self.cfg = config
        self.embed = SentenceTransformer("intfloat/multilingual-e5-small")
        # Removing reranker for faster/safer verification
        self.reranker = None 
        self.tok = sudachipy.Dictionary().create()
        client = chromadb.PersistentClient(path=db_path)
        self.tc = client.get_collection(f"ctd_{drug_id}_text")
        self.tbc = client.get_collection(f"ctd_{drug_id}_table")
        self.fc = client.get_collection(f"ctd_{drug_id}_figure")

        all_docs = self.tc.get()
        self.bm25_docs = all_docs["documents"]
        self.bm25_ids = all_docs["ids"]
        self.bm25_meta = all_docs["metadatas"]
        corpus = [[t.surface() for t in self.tok.tokenize(d)] for d in self.bm25_docs]
        self.bm25 = BM25Okapi(corpus)

    def retrieve(self, query: str, if_section: str = None) -> RetrievalContext:
        K = self.cfg.get("top_k_retrieve", 10)
        wf = {"if_hint": if_section} if if_section else None

        # Dense
        qe = self.embed.encode(f"query: {query}").tolist()
        dr = self.tc.query(query_embeddings=[qe], n_results=K, where=wf)
        
        # Sparse
        toks = [t.surface() for t in self.tok.tokenize(query)]
        bm25_raw = self.bm25.get_scores(toks)
        top_bm = np.argsort(bm25_raw)[::-1][:K]
        
        # Hybrid union
        dr_ids = dr["ids"][0] if dr["ids"] else []
        all_ids = list(set(dr_ids) | {self.bm25_ids[i] for i in top_bm if bm25_raw[i] > 0})
        if not all_ids: return RetrievalContext()
        
        cands = self.tc.get(ids=all_ids[:K])
        psg = list(zip(cands["ids"], cands["documents"], cands["metadatas"]))
        
        ctx = RetrievalContext()
        for _id, doc, meta in psg:
            ctx.text_passages.append(doc)
            for tid in json.loads(meta.get("linked_tables", "[]")):
                if tid not in ctx.tables:
                    try:
                        r = self.tbc.get(ids=[tid])
                        if r["documents"]: ctx.tables[tid] = r["documents"][0]
                    except: pass
        return ctx
