import fitz # PyMuPDF
import re, os, base64, ollama
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class FigureChunk:
    figure_id: str
    caption: str
    figure_type: str # "pk_profile", "km_curve", etc.
    vision_text: str # Vision-LLM
    image_path: str
    ctd_module: str
    page_num: int

class CTDFigureExtractor:
    def __init__(self, output_dir: str, config=None):
        self.output_dir = output_dir
        self.cfg = config or {}
        self.vision_model = self.cfg.get("llm", {}).get("vision_model", "moondream")
        self.host = self.cfg.get("llm", {}).get("ollama_host", "http://localhost:11434")
        self.PK_KW = ["plasma concentration","血中濃度","pk profile","pharmacokinetics"]
        self.KM_KW = ["kaplan-meier","survival","生存率"]
        self.DR_KW = ["dose response","用量反応","ic50","ed50"]
        self.CAP_PAT = re.compile(r"(?:図|Figure|Fig\.)\s*(\d+(?:\.\d+)*)\s*(.*?)(?=\n|$)", re.IGNORECASE)

    def classify(self, caption: str, vision_text: str) -> str:
        txt = (caption + " " + vision_text).lower()
        if any(k in txt for k in self.PK_KW): return "pk_profile"
        if any(k in txt for k in self.KM_KW): return "km_curve"
        if any(k in txt for k in self.DR_KW): return "dose_response"
        if "structure" in txt or "mol" in txt: return "structure"
        return "other"

    def analyze_with_vision_llm(self, image_path: str, caption: str) -> str:
        """Vision-LLM (moondream) でグラフ内容をテキスト化"""
        try:
            with open(image_path, "rb") as f:
                img_b64 = base64.b64encode(f.read()).decode()
            
            prompt = (f"Caption: {caption}\n"
                      "Extract the following from the image: 1. X-axis label and unit, 2. Y-axis label and unit, "
                      "3. Legend content, 4. Key numerical values, 5. Graph type.")
            
            # Using ollama directly via httpx if needed, or ollama library
            # Since I already have a 'client' in tests, I'll use the ollama library
            resp = ollama.chat(
                model=self.vision_model, 
                messages=[{"role":"user","content":prompt,"images":[img_b64]}])
            return resp["message"]["content"]
        except Exception as e:
            return f"Vision analytics skip: {e}"

    def extract(self, pdf_path: str, ctd_module: str) -> List[FigureChunk]:
        from pathlib import Path
        doc = fitz.open(str(pdf_path))
        target_dir = os.path.join(self.output_dir, Path(pdf_path).stem)
        Path(target_dir).mkdir(parents=True, exist_ok=True)
        
        all_text = " ".join(page.get_text() for page in doc)
        captions = {m.group(1): m.group(2).strip() for m in self.CAP_PAT.finditer(all_text)}
        
        chunks, fig_count = [], 0
        for page_num, page in enumerate(doc):
            for img in page.get_images(full=True):
                xref = img[0]
                pix = fitz.Pixmap(doc, xref)
                if pix.n > 4: pix = fitz.Pixmap(fitz.csRGB, pix)
                
                fig_count += 1
                fid = f"Figure_{fig_count}"
                img_path = os.path.join(target_dir, f"{fid}.png")
                pix.save(img_path)
                
                caption = captions.get(str(fig_count), f"Figure {fig_count}")
                vtext = self.analyze_with_vision_llm(img_path, caption)
                ftype = self.classify(caption, vtext)
                
                chunks.append(FigureChunk(
                    figure_id=fid, caption=caption,
                    figure_type=ftype, vision_text=vtext,
                    image_path=img_path, ctd_module=ctd_module,
                    page_num=page_num))
        return chunks
