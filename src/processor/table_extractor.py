import camelot, pdfplumber, json, re
from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class TableChunk:
    table_id: str
    title: str
    headers: List[str]
    rows: List[List[str]]
    footnotes: List[str]
    json_repr: str # LLM入力用JSON文字列
    ctd_module: str
    page_num: int
    method: str # "lattice" or "stream"
    accuracy: float

class CTDTableExtractor:
    def extract(self, pdf_path: str, ctd_module: str) -> List[TableChunk]:
        chunks = []
        # 方法1: Camelot lattice（格子型・罫線あり）
        try:
            tables = camelot.read_pdf(str(pdf_path), pages="all", flavor="lattice", line_scale=40)
            for i, t in enumerate(tables):
                if t.accuracy < 0.7: continue
                chunk = self._process_camelot(t, f"Table_L_{i}", ctd_module, "lattice")
                if chunk: chunks.append(chunk)
        except Exception as e:
            print(f"Camelot lattice error in {pdf_path}: {e}")

        # 方法2: pdfplumber (stream型・罫線なし)
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for i, page in enumerate(pdf.pages):
                    tbls = page.extract_tables()
                    for j, t in enumerate(tbls):
                        if not t or len(t) < 2: continue
                        chunk = self._process_plumber(t, f"Table_S_{i}_{j}", ctd_module, i)
                        if chunk: chunks.append(chunk)
        except Exception as e:
            print(f"pdfplumber error in {pdf_path}: {e}")
            
        return chunks

    def _process_camelot(self, table, tid, module, method) -> Optional[TableChunk]:
        df = table.df
        if df.empty or df.shape[0] < 2: return None
        headers = df.iloc[0].tolist()
        rows = df.iloc[1:].values.tolist()
        data = {"headers": [str(h) for h in headers], "rows": [[str(c) for c in r] for r in rows]}
        return TableChunk(
            table_id=tid, title=f"Table from {module}",
            headers=headers, rows=rows, footnotes=[],
            json_repr=json.dumps(data, ensure_ascii=False),
            ctd_module=module, page_num=table.page,
            method=method, accuracy=table.accuracy)

    def _process_plumber(self, table, tid, module, pnum) -> Optional[TableChunk]:
        headers = [str(h) for h in table[0] if h]
        rows = [[str(c) for c in r] for r in table[1:]]
        data = {"headers": headers, "rows": rows}
        return TableChunk(
            table_id=tid, title=f"Table from {module}",
            headers=headers, rows=rows, footnotes=[],
            json_repr=json.dumps(data, ensure_ascii=False),
            ctd_module=module, page_num=pnum,
            method="stream", accuracy=0.8)
