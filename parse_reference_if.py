import fitz, json, re
from pathlib import Path

def parse_if_pdf(pdf_path, output_json):
    doc = fitz.open(pdf_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    
    # Improved patterns using the actual characters seen in the text
    # Using \s* to handle potential spaces between numeral and title
    section_patterns = {
        "section_I": r"[ⅠI][．.].*概要",
        "section_II": r"[ⅡII][．.].*名称",
        "section_III": r"[ⅢIII][．.].*有効成分",
        "section_IV": r"[ⅣIV][．.].*製剤",
        "section_V": r"[ⅤV][．.].*治療",
        "section_VI": r"[ⅥVI][．.].*薬効薬理",
        "section_VII": r"[ⅦVII][．.].*薬物動態",
        "section_VIII": r"[ⅧVIII][．.].*安全性",
        "section_IX": r"[ⅨIX][．.].*非臨床試験",
        "section_X": r"[ⅩX][．.].*管理的事項",
        "section_XI": r"[ⅪXI][．.].*文献",
        "section_XII": r"[ⅫXII][．.].*参考資料",
        "section_XIII": r"[ⅩⅢXIII][．.].*備考",
    }
    
    sections = {}
    keys = list(section_patterns.keys())
    
    # First, find all start indices
    matches = []
    for key in keys:
        pattern = section_patterns[key]
        # We search from the beginning. Note: The Table of Contents also has these headers.
        # We should skip the ToC. ToC headers usually have ".... 1" at the end.
        # Modern IFs have headers as separate lines.
        all_matches = list(re.finditer(pattern, full_text))
        if all_matches:
            # Pick the last match if multiple found (usually the actual header after ToC)
            # Actually, ToC is at the start. So we pick the match that is NOT followed by dots and numbers.
            # Or just filter out matches that appear in the first ~5000 chars (ToC area).
            valid_match = None
            for m in all_matches:
                if m.start() > 3000: # Skip ToC roughly
                    valid_match = m
                    break
            
            if valid_match:
                matches.append((key, valid_match.start(), valid_match.end()))
    
    # Sort matches by position
    matches.sort(key=lambda x: x[1])
    
    for i in range(len(matches)):
        curr_key, start_pos, content_start = matches[i]
        
        # End is start of next section
        if i + 1 < len(matches):
            end_pos = matches[i+1][1]
        else:
            end_pos = len(full_text)
            
        sections[curr_key] = full_text[content_start:end_pos].strip()
        
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(sections, f, ensure_ascii=False, indent=2)
    print(f"Parsed {len(sections)} sections and saved to {output_json}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Parse IF PDF into sections JSON")
    parser.add_argument("--target_dir", type=str, required=True, help="Directory containing the drug data")
    parser.add_argument("--reference_file", type=str, required=True, help="Filename of the reference PDF in target_dir")
    args = parser.parse_args()
    
    ref_pdf = Path(args.target_dir) / args.reference_file
    output_json = Path(args.target_dir) / "if_human_parsed.json"
    
    if ref_pdf.exists():
        parse_if_pdf(str(ref_pdf), str(output_json))
    else:
        print(f"Ref PDF not found: {ref_pdf}")

