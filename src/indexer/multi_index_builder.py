import chromadb, json
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

class MultiIndexBuilder:
    def __init__(self, db_path: str, drug_id: str):
        # Changed from 'multilingual-e5-large' to 'multilingual-e5-small' for faster local CPU indexing
        self.embed = SentenceTransformer("intfloat/multilingual-e5-small")
        self.client = chromadb.PersistentClient(path=db_path)
        self.tc = self.client.get_or_create_collection(
            f"ctd_{drug_id}_text", metadata={"hnsw:space": "cosine"})
        self.tbc = self.client.get_or_create_collection(
            f"ctd_{drug_id}_table", metadata={"hnsw:space": "cosine"})
        self.fc = self.client.get_or_create_collection(
            f"ctd_{drug_id}_figure", metadata={"hnsw:space": "cosine"})

    def index_text(self, chunks, batch=128): # Batch size increased to 128 for 'small' model
        for i in range(0, len(chunks), batch):
            b = chunks[i:i+batch]
            docs, ids, metas = [], [], []
            for c in b:
                for j, child in enumerate(c.child_texts):
                    docs.append(child)
                    ids.append(f"{c.chunk_id}_c{j}")
                    metas.append({
                        "parent_id": c.chunk_id,
                        "ctd_module": c.ctd_module,
                        "section": c.section_num,
                        "if_hint": c.if_section_hint,
                        "language": c.language,
                        "page": c.page_num,
                        "linked_tables": json.dumps(c.linked_table_ids),
                        "linked_figures": json.dumps(c.linked_figure_ids)})
            
            embs = self.embed.encode([f"passage: {d}" for d in docs]).tolist()
            self.tc.upsert(documents=docs, embeddings=embs, ids=ids, metadatas=metas)
            print(f"Text indexed: {min(i+batch, len(chunks))}/{len(chunks)}")

    def index_tables(self, tables):
        for t in tables:
            txt = f"passage: {t.title} {' '.join(t.headers)}"
            emb = self.embed.encode([txt]).tolist()
            self.tbc.upsert(documents=[t.json_repr], embeddings=emb, ids=[t.table_id],
                           metadatas=[{"ctd_module": t.ctd_module, "page": t.page_num,
                                      "accuracy": t.accuracy, "method": t.method}])

    def index_figures(self, figures):
        for f in figures:
            txt = f"passage: {f.caption} {f.vision_text}"
            emb = self.embed.encode([txt]).tolist()
            self.fc.upsert(documents=[f.caption + " " + f.vision_text],
                           embeddings=emb, ids=[f.figure_id],
                           metadatas=[{"ctd_module": f.ctd_module, "type": f.figure_type,
                                      "page": f.page_num, "image_path": f.image_path}])
