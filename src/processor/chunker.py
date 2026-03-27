from dataclasses import dataclass, field
from typing import List, Optional
import hashlib

@dataclass
class ContextualChunk:
    chunk_id: str
    parent_text: str # 親チャンク (~1024 tok)
    child_texts: List[str] # 子チャンク (~256 tok each)
    linked_table_ids: List[str] # 参照される表ID (テキストに直結)
    linked_figure_ids:List[str] # 参照される図ID (テキストに直結)
    ctd_module: str
    section_num: str
    if_section_hint: str # "section_VII" 等
    language: str
    page_num: int

class ContextualChunker:
    # CTDモジュール → IFセクションヒント
    IF_HINT_MAP = {
        "Module_2.7.1": "section_VII", "Module_2.7.2": "section_VII",
        "Module_2.7.3": "section_V",   "Module_2.7.4": "section_VIII",
        "Module_2.6.2": "section_VI",  "Module_3.2.P": "section_IV",
        "Module_4.2.3": "section_IX",  "Module_4.2.1": "section_VI",
        "Module_3.2.S": "section_III", "Module_2.3": "section_IV",
        "Module_2.5": "section_I", 
    }

    def __init__(self, parent_size=1200, child_size=300):
        self.ps = parent_size
        self.cs = child_size

    def chunk(self, text_blocks, table_chunks=None, figure_chunks=None) -> List[ContextualChunk]:
        chunks, buf, rt, rf = [], "", [], []
        last_blk = None
        for blk in text_blocks:
            buf += blk.text + " "
            rt.extend(blk.ref_table_ids)
            rf.extend(blk.ref_figure_ids)
            last_blk = blk
            if len(buf) >= self.ps:
                chunks.append(self._make(buf, rt, rf, last_blk))
                buf, rt, rf = "", [], []
        
        if buf.strip():
            chunks.append(self._make(buf, rt, rf, last_blk))
        return chunks

    def _make(self, text, rt, rf, blk):
        cid = hashlib.md5(text.encode()).hexdigest()[:16]
        children = [text[i:i+self.cs] for i in range(0, len(text), self.cs)]
        m = blk.ctd_module if blk else "Unknown"
        return ContextualChunk(
            chunk_id=cid, parent_text=text.strip(),
            child_texts=children,
            linked_table_ids=list(set(rt)),
            linked_figure_ids=list(set(rf)),
            ctd_module=m,
            section_num=blk.section_num or "" if blk else "",
            if_section_hint=self.IF_HINT_MAP.get(m, "section_UNKNOWN"),
            language=blk.language.value if blk else "unknown",
            page_num=blk.page_num if blk else 0)
