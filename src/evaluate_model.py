import os, sys, json, pickle, glob, re
import numpy as np
import pandas as pd
from tqdm import tqdm
import xgboost as xgb
from gensim.models import KeyedVectors
from preprocessing import (
    preprocess_pdf_enhanced,
    extract_enhanced_features_from_preprocessed_data,
    strip_dot_leader,
    is_dot_leader,
    collapse_exactly4_repeats,
    _contains_multiple_numberings,
)

try:
    from rapidfuzz.fuzz import token_set_ratio
except ImportError:
    try:
        from fuzzywuzzy.fuzz import token_set_ratio
    except ImportError:
        print("Warning: Neither rapidfuzz nor fuzzywuzzy found. Installing rapidfuzz...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "rapidfuzz"])
        from rapidfuzz.fuzz import token_set_ratio

MODELFILE = "/app/model/heading_steroid_xgb.json"
SCALER_FILE = "/app/model/feature_scaler.pkl"
LABELS_FILE = "/app/model/heading_steroid_xgb.json.labels.json"
FASTTEXT_VEC = "/app/model/wiki-news-300d-subword-50k.vec"
USE_EMBEDDINGS = True

FEATURES = [
    "font_size","font_size_relative","font_size_zscore","is_largest_font","is_all_caps","is_title_case",
    "starts_with_number","text_length","word_count","has_colon","is_bold","is_upright",
    "page_num","x0","x1","y0","y1","width","height",
    "x_relative","y_relative","is_left_aligned","is_centered","is_top_area",
    "contains_heading_keywords","has_section_number","looks_like_heading",
    "is_dot_leader", "fraction_uppercase", "percent_alpha", "ends_with_colon", "is_very_short_heading_candidate"
]

def dedupe_headings(outline, title):
    seen = set()
    new_outline = []
    for h in outline:
        key = (h['level'], h['text'].strip().lower())
        if h['text'].strip().lower() == title.strip().lower():
            continue
        if key in seen: 
            continue
        seen.add(key)
        new_outline.append(h)
    return new_outline

def strip_repeated_colons(txt):
    return re.sub(r'[:：\·]{2,}', ':', txt).strip()

model = xgb.XGBClassifier()
model.load_model(MODELFILE)
with open(SCALER_FILE, "rb") as f: 
    scaler = pickle.load(f)
with open(LABELS_FILE, "r") as f: 
    label_classes = json.load(f)
WV_MODEL = KeyedVectors.load_word2vec_format(FASTTEXT_VEC, binary=False)
EMB_DIM = WV_MODEL.vector_size

def sentence_embedding(text: str) -> np.ndarray:
    words = [w.lower().strip() for w in text.split() if w.strip()]
    if not words: 
        return np.zeros(EMB_DIM)
    vectors = [WV_MODEL[w] for w in words if w in WV_MODEL]
    if vectors: 
        return np.mean(vectors, axis=0)
    else: 
        return np.zeros(EMB_DIM)

def merge_heading_blocks_all_levels(headings, df):
    order = ['H1','H2','H3','H4','H5','H6']
    merged = []
    i = 0
    while i < len(headings):
        base = headings[i]
        page = base["page"]
        texts = [base["text"]]
        levels = [base["level"]]
        match_df = df[(df.page_num == page) & (df.text == base["text"])]
        base_y = match_df['y0'].iloc[0] if not match_df.empty else 0
        base_font = match_df['font_size'].iloc[0] if not match_df.empty else 12

        # *** Only merge next block if it does NOT look like another heading number ***
        j = i + 1
        while j < len(headings):
            next_ = headings[j]
            if next_["page"] != page:
                break
            # If next heading text starts with a section number, break! do NOT merge
            if re.match(r'^\d', next_["text"].strip()):
                break
            next_df = df[(df.page_num == page) & (df.text == next_["text"])]
            next_y = next_df['y0'].iloc[0] if not next_df.empty else base_y
            next_font = next_df['font_size'].iloc[0] if not next_df.empty else base_font
            if abs(next_y - base_y) < (4 * base_font) and abs(next_font - base_font) < 2:
                texts.append(next_["text"])
                levels.append(next_["level"])
                base_y, base_font = next_y, next_font
                j += 1
            else:
                break
        block_level = sorted(set(levels), key=lambda x: order.index(x) if x in order else 99)[0]
        block_text = ' '.join(texts)
        merged.append({"level": block_level, "text": block_text, "page": page})
        i = j
    return merged

def extract_best_title(df, outline):
    first_page = df[df.page_num == 1]
    if not first_page.empty:
        idxmax = first_page['font_size'].idxmax()
        row = first_page.loc[idxmax]
        y0 = row['y0']
        font_size = row['font_size']
        lines = first_page[
            (abs(first_page['font_size'] - font_size) < 1.5) &
            (abs(first_page['y0'] - y0) < 3.0 * font_size)
        ].sort_values('y0')
        text = ' '.join([strip_repeated_colons(str(t)).strip() for t in lines['text']])
        text = collapse_exactly4_repeats(text)
        return text.strip()
    return "Untitled Document"

def pdf_to_outline(pdf_path):
    page_data = preprocess_pdf_enhanced(pdf_path)
    feats = extract_enhanced_features_from_preprocessed_data(page_data)
    if not feats:
        print(f"[{os.path.basename(pdf_path)}] No text found.")
        return {"title": "No Content Found", "outline": []}
    df = pd.DataFrame(feats)
    df.text = df.text.apply(collapse_exactly4_repeats)
    
    X = scaler.transform(df[FEATURES])
    if USE_EMBEDDINGS:
        emb = np.vstack([sentence_embedding(t) for t in df.text])
        X = np.hstack([X, emb])
    labels = model.predict(X)
    df["pred_label"] = [label_classes[int(i)] for i in labels]
    outline = []
    for _, row in df.iterrows():
        pred = row["pred_label"]
        if pred == "not-heading": 
            continue
        cleaned_heading = strip_dot_leader(row["text"])
        cleaned_heading = collapse_exactly4_repeats(cleaned_heading)
        outline.append({
            "level": pred,
            "text": cleaned_heading,
            "page": int(row["page_num"])
        })
    
    outline = sorted(outline, key=lambda x: (x["page"],
        df[(df.text == x['text']) & (df.page_num == x['page'])].iloc[0]['y0'] 
        if not df[(df.text == x['text']) & (df.page_num == x['page'])].empty else 0))
    
    merged_outline = merge_heading_blocks_all_levels(outline, df)
    document_title = extract_best_title(df, merged_outline)
    
    n_pages = len(page_data)
    title_page = None
    title_text_lower = collapse_exactly4_repeats(document_title.strip().lower())
    
    for h in merged_outline:
        h_text_lower = collapse_exactly4_repeats(h["text"].strip().lower())
        if h_text_lower == title_text_lower:
            title_page = h["page"]
            break
    
    final_outline = []
    for item in merged_outline:
        item_text_lower = collapse_exactly4_repeats(item['text'].strip().lower())
        
        title_similarity = token_set_ratio(item_text_lower, title_text_lower)
        
        # Enhanced title fragment detection
        if (title_similarity >= 75 or 
            item_text_lower.startswith(title_text_lower[:10]) or
            any(word in title_text_lower.split() for word in item_text_lower.split() if len(word) > 3) and
            len(set(title_text_lower.split()) & set(item_text_lower.split())) >= min(3, len(item_text_lower.split()))):
            continue
        
        if n_pages > 3  and item['page'] == 1:
            continue
            
        final_outline.append(item)
    
    final_outline_simple = [
        {k: h[k] for k in ("level", "text", "page")}
        for h in final_outline
    ]
    return {"title": document_title, "outline": final_outline_simple}

if __name__ == "__main__":
    import sys, os, glob, json
    from tqdm import tqdm

    pdf_files = []
    for arg in sys.argv[1:-1]:  # Optional: if you take input folder(s) first and output dir last
        if os.path.isdir(arg):
            pdf_files.extend(glob.glob(os.path.join(arg, "*.pdf")))
        else:
            pdf_files.append(arg)
    if not pdf_files:
        pdf_files = glob.glob("*.pdf")
    if not pdf_files:
        print("No PDFs found. Usage: python evaluate_model.py [file.pdf folder ...] <output_dir>")
        sys.exit(1)
    output_dir = sys.argv[-1] if len(sys.argv) > 1 else "./"  # get output dir, default cwd
    os.makedirs(output_dir, exist_ok=True)
    print(f"Processing {len(pdf_files)} PDF(s)...")
    for pdf_path in tqdm(pdf_files, desc="Processing PDFs"):
        try:
            result = pdf_to_outline(pdf_path)
            base = os.path.splitext(os.path.basename(pdf_path))[0]
            output_file = os.path.join(output_dir, f"{base}.json")
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=4, ensure_ascii=False)
            print(f"✓ {pdf_path} -> {output_file} ({len(result['outline'])} heading blocks)")
        except Exception as e:
            print(f"✗ Error processing {pdf_path}: {e}")
    print("X===All Files Processed===X")

