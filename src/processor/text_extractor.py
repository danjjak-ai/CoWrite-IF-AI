import fitz # PyMuPDF
import re
from dataclasses import dataclass, field
from typing import List, Optional
from enum import Enum

class Language(Enum):
    JAPANESE = "ja"
    ENGLISH = "en"
    MIXED = "mixed"

@dataclass
class TextBlock:
    text: str
    language: Language
    is_heading: bool
    section_num: Optional[str] # "2.7.2.1" など CTDセクション番号
    ctd_module: str # "Module_2.7.2" など
    page_num: int
    bbox: tuple
    font_size: float
    is_bold: bool
    ref_table_ids: List[str] = field(default_factory=list) # 「表N」参照
    ref_figure_ids: List[str] = field(default_factory=list) # 「図N」参照

class CTDTextExtractor:
    # CTDモジュールとファイル名パターンの対応
    CTD_MODULE_PATTERNS = {
        "Module_2.3": ["2_3","m23","quality_summary"],
        "Module_2.6.2": ["2_6_2","pharm_summary","pharmacology_summary"],
        "Module_2.7.1": ["2_7_1","biopharm","ba_study","bioavailability"],
        "Module_2.7.2": ["2_7_2","clinpharm","clinical_pharmacology"],
        "Module_2.7.3": ["2_7_3","efficacy","clinical_efficacy"],
        "Module_2.7.4": ["2_7_4","safety","clinical_safety"],
        "Module_3.2.P": ["3_2_p","formulation","pharmaceutical"],
        "Module_4.2.1": ["4_2_1","pharm_studies","pharmacology_studies"],
        "Module_4.2.3": ["4_2_3","tox","toxicology"],
        "Module_5.3": ["5_3","clinical_study"],
    }
    REF_TABLE_PAT = re.compile(r"(?:表|Table)\s*(\d+(?:\.\d+)*)", re.IGNORECASE)
    REF_FIGURE_PAT = re.compile(r"(?:図|Figure|Fig\.)\s*(\d+(?:\.\d+)*)", re.IGNORECASE)
    SECTION_PAT = re.compile(r"^(\d+\.\d+(?:\.\d+)*)")
    JA_PAT = re.compile(r"[\u3040-\u30ff\u4e00-\u9fff]")

    def detect_module(self, filename: str) -> str:
        fn = filename.lower()
        for module, patterns in self.CTD_MODULE_PATTERNS.items():
            if any(p in fn for p in patterns): return module
        
        # ファイル名に明確なパターンがない場合、適応的に推論
        if "_b" in fn: return "Module_2.3"
        if "_d" in fn: return "Module_2.6"
        if "_e" in fn: return "Module_2.7.1"
        if "_f" in fn: return "Module_2.7.2"
        if "_g" in fn: return "Module_2.7.3"
        if "_h" in fn: return "Module_2.7.4"
        if "_i" in fn: return "Module_4"
        if "_j" in fn: return "Module_5"
        
        return "Module_UNKNOWN"

    def detect_language(self, text: str) -> Language:
        if not text: return Language.ENGLISH
        ratio = len(self.JA_PAT.findall(text)) / max(len(text), 1)
        if ratio > 0.3: return Language.JAPANESE
        if ratio > 0.05: return Language.MIXED
        return Language.ENGLISH

    def extract_refs(self, text: str):
        tables = [f"Table_{m}" for m in self.REF_TABLE_PAT.findall(text)]
        figures = [f"Figure_{m}" for m in self.REF_FIGURE_PAT.findall(text)]
        return tables, figures

    def extract(self, pdf_path: str) -> List[TextBlock]:
        from pathlib import Path
        module = self.detect_module(Path(pdf_path).name)
        doc = fitz.open(str(pdf_path))
        blocks = []
        for page_num, page in enumerate(doc):
            raw = page.get_text("dict")["blocks"]
            for blk in raw:
                if blk.get("type") != 0: continue
                spans = [s for line in blk.get("lines",[]) for s in line.get("spans",[])]
                if not spans: continue
                text = " ".join(s["text"].strip() for s in spans if s["text"].strip())
                if len(text) < 3: continue
                
                fs = spans[0]["size"]
                bold = "bold" in spans[0]["font"].lower()
                sm = self.SECTION_PAT.search(text)
                rt, rf = self.extract_refs(text)
                
                blocks.append(TextBlock(
                    text=text, language=self.detect_language(text),
                    is_heading=(fs > 11 or bold),
                    section_num=sm.group(1) if sm else None,
                    ctd_module=module, page_num=page_num,
                    bbox=blk["bbox"], font_size=fs, is_bold=bold,
                    ref_table_ids=rt, ref_figure_ids=rf))
        return blocks
