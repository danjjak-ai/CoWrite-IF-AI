import os, yaml, json
from pathlib import Path
from src.processor.text_extractor import CTDTextExtractor
from src.processor.table_extractor import CTDTableExtractor
from src.processor.vision_analyzer import CTDFigureExtractor
from src.processor.chunker import ContextualChunker
from src.indexer.multi_index_builder import MultiIndexBuilder
from src.rag.retriever import CrossModalRAGPipeline

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--drug", help="Drug ID overriding config.yaml")
    args = parser.parse_args()

    # 1. Load Config
    with open("config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    
    drug_id = args.drug or cfg["system"]["drug_id"]
    pdf_dir = Path(f"data/raw/{drug_id}/ctd")
    out_dir = Path(f"data/processed/{drug_id}")
    db_path = f"data/vectordb/{drug_id}"
    
    pdfs = list(pdf_dir.glob("*.pdf"))
    if not pdfs:
        print("No PDFs found!")
        return

    # 2. Initialize Components
    text_ext = CTDTextExtractor()
    table_ext = CTDTableExtractor()
    fig_ext = CTDFigureExtractor(str(out_dir / "figures"), config=cfg)
    chunker = ContextualChunker()
    builder = MultiIndexBuilder(db_path, drug_id)

    all_blocks = []
    all_tables = []
    all_figures = []

    # 3. Step 2 & 3: Extraction
    for pdf in pdfs:
        print(f"--- Processing {pdf.name} ---")
        module = text_ext.detect_module(pdf.name)
        
        # Step 2: Text
        blocks = text_ext.extract(str(pdf))
        all_blocks.extend(blocks)
        print(f"  Extracted {len(blocks)} text blocks")
        
        # Step 3: Tables
        tables = table_ext.extract(str(pdf), module)
        all_tables.extend(tables)
        print(f"  Extracted {len(tables)} tables")
        
        # Step 3: Figures (Vision)
        # We limit to first 3 figures per PDF if they are many, to save time/vision tokens
        # But here we just extract all as per prompt
        try:
            figs = fig_ext.extract(str(pdf), module)
            all_figures.extend(figs)
            print(f"  Extracted {len(figs)} figures")
        except Exception as e:
            print(f"  Figure extraction error: {e}")

    # 4. Step 4: Chunking & Indexing
    print("--- Chunking and Indexing ---")
    chunks = chunker.chunk(all_blocks)
    print(f"  Created {len(chunks)} contextual chunks")
    
    builder.index_text(chunks)
    builder.index_tables(all_tables)
    builder.index_figures(all_figures)
    print("  Indexing complete.")

    # 5. Step 5: Verification
    print("--- Verifying RAG Pipeline ---")
    try:
        pipe = CrossModalRAGPipeline(db_path, drug_id, cfg["rag"])
        query = "薬物動態 パラメータ Cmax AUC"
        ctx = pipe.retrieve(query, if_section="section_VII")
        
        print(f"  Query: {query}")
        print(f"  Text Results: {len(ctx.text_passages)}")
        print(f"  Tables Found: {list(ctx.tables.keys())}")
        print(f"  Figures Found: {list(ctx.figures.keys())}")
        
        # Save summary
        summary = {
            "total_blocks": len(all_blocks),
            "total_tables": len(all_tables),
            "total_figures": len(all_figures),
            "total_chunks": len(chunks)
        }
        with open(out_dir / "pipeline_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
            
    except Exception as e:
        print(f"  RAG Verification Error: {e}")

if __name__ == "__main__":
    main()
