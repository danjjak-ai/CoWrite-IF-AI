import ollama, yaml, json
from pathlib import Path
from src.rag.retriever import CrossModalRAGPipeline

IF_SECTIONS = {
    "section_I": {"name":"Ⅰ. 概要に関する項目", "ctd":"2.5", "tokens":3000},
    "section_II": {"name":"Ⅱ. 名称に関する項目", "ctd":"3.2.S", "tokens":1000},
    "section_III": {"name":"Ⅲ. 有効成分に関する項目", "ctd":"3.2.S", "tokens":2000},
    "section_IV": {"name":"Ⅳ. 製剤に関する項目", "ctd":"3.2.P", "tokens":3000},
    "section_V": {"name":"Ⅴ. 治療に関する項目", "ctd":"2.7.3", "tokens":6000},
    "section_VI": {"name":"Ⅵ. 薬効薬理に関する項目", "ctd":"2.6.2", "tokens":3000},
    "section_VII": {"name":"Ⅶ. 薬物動態に関する項目", "ctd":"2.7.2", "tokens":5000},
    "section_VIII": {"name":"Ⅷ. 安全性に関する項目", "ctd":"2.7.4", "tokens":4000},
    "section_IX": {"name":"Ⅸ. 非臨床試験に関する項目","ctd":"4.2", "tokens":4000},
    "section_X": {"name":"Ⅹ. 管理的事項に関する項目","ctd":"1.2", "tokens":2000},
    "section_XI": {"name":"ⅩⅠ. 文献", "ctd":"all", "tokens":1000},
    "section_XII": {"name":"ⅩⅡ. 参考資料", "ctd":"all", "tokens":1500},
    "section_XIII": {"name":"ⅩⅢ. 備考（調剤・服薬支援含む）","ctd":"3.2.P","tokens":2000},
}

class IFSectionGenerator:
    def __init__(self, drug_id, drug_name, rag: CrossModalRAGPipeline, 
                 model="qwen2.5-coder:7b", prompt_dir="./prompts", 
                 host="http://localhost:11434"):
        self.drug_id = drug_id
        self.drug_name = drug_name
        self.rag = rag
        self.model = model
        self.prompt_dir = Path(prompt_dir)
        self.host = host

    def load_prompt(self, key: str) -> dict:
        for base in ["tuned", "base"]:
            p = self.prompt_dir / base / f"{key}.yaml"
            if p.exists():
                with open(p, encoding="utf-8") as f: return yaml.safe_load(f)
        return self._default_prompt(key)

    def _default_prompt(self, key: str) -> dict:
        sec = IF_SECTIONS.get(key, {})
        return {
            "system": """あなたはIF記載要領2018（2019年更新版）に完全準拠した
インタビューフォーム作成の専門家です。
【必須ルール】
1. 数値は参照データから1桁も変えずに転記すること
2. 表が言言及されている場合は「表N」の形式で本文中に参照すること
3. 図が言及されている場合は「図N」の形式で本文中に参照すること
4. 単位は元データの単位をそのまま使用すること
5. 専門用語を正確に使用し、学術文体（〜である）で書くこと""",
            "user_template": f"医薬品「{{drug_name}}」의 {sec.get('name',key)}을(를) 다음 참조 데이터를 바탕으로 작성해줘:\n{{context}}"
        }

    def generate_section(self, key: str) -> str:
        sec = IF_SECTIONS.get(key, {})
        query = f"{self.drug_name} {sec.get('ctd','')} {sec.get('name',key)}"
        ctx = self.rag.retrieve(query, if_section=key)
        tmpl = self.load_prompt(key)
        
        user_prompt = tmpl["user_template"].format(
            drug_name=self.drug_name, 
            context=ctx.to_llm_input())
            
        from ollama import Client
        oa_client = Client(host=self.host)
        
        resp = oa_client.chat(
            model=self.model,
            messages=[{"role":"system","content":tmpl["system"]},
                      {"role":"user","content":user_prompt}],
            options={"temperature":0.1, "num_predict":sec.get("tokens",3000), "num_ctx":8192})
        return resp["message"]["content"]

    def generate_full_if(self) -> dict:
        result = {}
        for key, cfg in IF_SECTIONS.items():
            print(f" Generating {cfg['name']}...")
            result[key] = self.generate_section(key)
        return result
