import os
import pdfplumber
import re
import numpy as np
from typing import List, Dict
from collections import Counter, defaultdict
from difflib import SequenceMatcher

MARGIN_FRAC = 0.08

def line_y_tol(fs):
    return max(16.0, 1.6 * fs)


def collapse_exactly4_repeats(text: str) -> str:
    return re.sub(r'(.)\1{3}(?!\1)', r'\1', text)

def is_dot_leader(txt: str) -> bool:
    dots = re.sub(r'[^\.\·]', '', txt)
    return len(txt) > 6 and len(dots) > 0.35 * len(txt)

def strip_dot_leader(txt: str) -> str:
    return re.sub(r'[ \.\·]{5,}', ' ', txt).strip()

def collapse_repeated_colons(txt: str) -> str:
    return re.sub(r'[:：\·]{2,}', ':', txt).strip()

def detect_bold_text(fontname: str, text: str) -> bool:
    if any(ind in fontname.lower() for ind in ['bold', 'heavy', 'black']):
        return True
    if re.search(r'(.)\1{2,}', text):
        return True
    return False


def _contains_multiple_numberings(text: str) -> bool:
    patterns = [
        r'\b\d+\.\d+\b',
        r'\bChapter\s+\d+\b',
        r'\bSection\s+\d+\b',
        r'^\d+\.\s',
        r'\b\d+\)\s',
    ]
    matches = 0
    for pattern in patterns:
        if re.search(pattern, text, re.IGNORECASE):
            matches += 1
            if matches >= 2:
                return True
    return False

def heading_starts(words):
    indexes = []
    for i, t in enumerate(words):
        if re.match(r'^\d+(\.\d+)*(\.|[)]| )', t.strip()):  # e.g. 1. , 1.1, 3)
            indexes.append(i)
        elif i == 0 and t[0].isalpha() and t[0].isupper():
            indexes.append(i)  # fallback: first capital word at begin
    return indexes

def is_structural_table(table) -> bool:
    try:
        if hasattr(table, 'rows'):
            rows = len(table.rows) if table.rows else 0
            cols = max(len(r.cells) for r in table.rows) if table.rows else 0
        else:
            rows = len(table.cells) if hasattr(table, 'cells') and isinstance(table.cells, list) else 0
            cols = len(table.columns) if hasattr(table, 'columns') else 0
        return (rows >= 2) and (cols >= 2)
    except Exception:
        return False

def exclude_words_in_tables(words, page):
    tables = page.find_tables()
    if not tables:
        return words
    cleaned = []
    for w in words:
        w_cen_x = (w["x0"] + w["x1"]) / 2
        w_cen_y = (w["top"] + w["bottom"]) / 2
        inside = False
        for t in tables:
            if (t.bbox[0] <= w_cen_x <= t.bbox[2] and t.bbox[1] <= w_cen_y <= t.bbox[3]):
                if is_structural_table(t):
                    inside = True
                    break
        if not inside:
            cleaned.append(w)
    return cleaned

def normalize_text_for_frequency(text: str) -> str:
    normalized = re.sub(r'\d+', '@NUM@', text.strip().lower())
    normalized = ' '.join(normalized.split())
    normalized = re.sub(r'[.,;:!?(){}[\]"\'-]', '', normalized)
    return normalized

def fuzzy_similarity(s1: str, s2: str) -> float:
    return SequenceMatcher(None, s1, s2).ratio()

try:
    from rapidfuzz.fuzz import token_set_ratio
except ImportError:
    from fuzzywuzzy.fuzz import token_set_ratio

def detect_frequency_based_headers_footers(all_page_data: List[Dict], n_pages: int) -> Dict:
    if n_pages <= 3:
        return {'headers': [], 'footers': []}
    page_lines = defaultdict(list)
    for page_data in all_page_data:
        page_num = page_data["page"]
        page_height = page_data["page_height"]
        lines = page_data["lines"]
        for i, line in enumerate(lines):
            text = line["text"].strip()
            if len(text) < 3:
                continue
            y_pos = line["top"]
            normalized = normalize_text_for_frequency(text)
            is_header_area = y_pos < page_height * 0.20
            is_footer_area = y_pos > page_height * 0.80
            if is_header_area or is_footer_area:
                page_lines[page_num].append((text, y_pos, normalized, is_header_area))
                # add adjacent pairs and triplets
                for addn in [1, 2]:
                    if i + addn < len(lines):
                        combo = text
                        for k in range(1, addn + 1):
                            combo += " " + lines[i + k]["text"].strip()
                        combo_norm = normalize_text_for_frequency(combo)
                        page_lines[page_num].append((combo, y_pos, combo_norm, is_header_area))
    header_candidates = []
    footer_candidates = []
    for page_num, lines in page_lines.items():
        for text, y_pos, normalized, is_header_area in lines:
            if is_header_area:
                header_candidates.append((page_num, y_pos, normalized, text))
            else:
                footer_candidates.append((page_num, y_pos, normalized, text))
    def cluster_similar_texts(texts_with_info, similarity_threshold=0.8):
        clusters = []
        for text_info in texts_with_info:
            text = text_info[2]
            added = False
            for c in clusters:
                if fuzzy_similarity(text, c[0][2]) >= similarity_threshold:
                    c.append(text_info)
                    added = True
                    break
            if not added:
                clusters.append([text_info])
        return clusters
    header_clusters = cluster_similar_texts(header_candidates, similarity_threshold=0.8)
    footer_clusters = cluster_similar_texts(footer_candidates, similarity_threshold=0.8)
    min_frequency = max(2, int(n_pages * 0.09))
    detected_headers = []
    detected_footers = []
    for cluster in header_clusters:
        if len(cluster) >= min_frequency:
            avg_y = np.mean([item[1] for item in cluster])
            normalized_texts = [item[2] for item in cluster]
            most_common_normalized = Counter(normalized_texts).most_common(1)[0][0]
            detected_headers.append({
                'normalized_text': most_common_normalized,
                'y_range': (avg_y - 40, avg_y + 40),
                'frequency': len(cluster),
                'example_text': cluster[0][3]
            })
    for cluster in footer_clusters:
        if len(cluster) >= min_frequency:
            avg_y = np.mean([item[1] for item in cluster])
            normalized_texts = [item[2] for item in cluster]
            most_common_normalized = Counter(normalized_texts).most_common(1)[0][0]
            detected_footers.append({
                'normalized_text': most_common_normalized,
                'y_range': (avg_y - 40, avg_y + 40),
                'frequency': len(cluster),
                'example_text': cluster[0][3]
            })
    return {'headers': detected_headers, 'footers': detected_footers}

def should_remove_as_header_footer(line_text: str, line_y: float, header_footer_patterns: Dict) -> bool:
    normalized = normalize_text_for_frequency(line_text)
    for kind in ['headers', 'footers']:
        for pat in header_footer_patterns[kind]:
            in_y = pat['y_range'][0] <= line_y <= pat['y_range'][1]
            sim = fuzzy_similarity(normalized, pat['normalized_text'])
            tsim = token_set_ratio(normalized, pat['normalized_text'])
            if in_y and (sim >= 0.8 or tsim >= 80):
                return True
    return False

def smart_split_cluster(cluster_sorted, page_width):
    if len(cluster_sorted) <= 1:
        return [cluster_sorted]
    words = [w["text"].strip() for w in cluster_sorted]
    # split on heading patterns
    idxs = heading_starts(words)
    # allow split if there's >1 heading-like index and not just index 0
    if len(idxs) > 1:
        segments=[]
        for k in range(len(idxs)):
            start = idxs[k]
            end = idxs[k+1] if k+1 < len(idxs) else len(words)
            segments.append(cluster_sorted[start:end])
        # merge any orphaned tiny right-most clusters
        min_len = 2
        for i in range(len(segments)-1, 0, -1):
            if len(segments[i]) < min_len and len(segments[i-1]) > 0:
                segments[i-1].extend(segments[i])
                segments[i]=[]
        return [seg for seg in segments if seg]
    # else fallback to big visual gaps (same as before)
    # Also handle too wide clusters
    def needs_split(cl):
        if len(cl) <= 1:
            return False
        span = cl[-1]["x1"] - cl[0]["x0"]
        merged_txt = ' '.join(w["text"].strip() for w in cl)
        return (span > 0.42 * page_width or 
                _contains_multiple_numberings(merged_txt) or
                merged_txt.count(':') > 1)
    segments = [cluster_sorted]
    # iterative visual split
    changed = True
    while changed:
        changed = False
        new_segments=[]
        for seg in segments:
            if needs_split(seg):
                gaps = [(b["x0"]-a["x1"], idx) for idx, (a,b) in enumerate(zip(seg, seg[1:]))]
                if not gaps:
                    new_segments += [seg]
                    continue
                max_gap, split_idx = max(gaps)
                median_width = np.median([w["x1"]-w["x0"] for w in seg])
                if max_gap > max(0.22 * page_width, 2*median_width):
                    left = seg[:split_idx+1]
                    right = seg[split_idx+1:]
                    if len(right) > 0:
                        new_segments += [left, right]
                    else:
                        new_segments += [left]
                    changed = True
                else:
                    new_segments += [seg]
            else:
                new_segments += [seg]
        segments = new_segments
    segments = [seg for seg in segments if seg]
    # merge any last single-token right chunks
    if len(segments) > 1 and len(segments[-1]) == 1:
        segments[-2].extend(segments[-1])
        segments = segments[:-1]
    return segments

def group_words_into_lines(
    words: List[Dict], page_width: float, page_height: float, n_pages: int, 
    header_footer_patterns: Dict = None
) -> List[Dict]:
    if n_pages > 3 and header_footer_patterns is None:
        words = [
            w for w in words if not (
                w['top'] < page_height * MARGIN_FRAC or w['bottom'] > page_height * (1 - MARGIN_FRAC)
            )
        ]
    if not words:
        return []

    words_sorted = sorted(words, key=lambda w: (w["top"], w["x0"]))

    # Step 1: group words by y into raw lines using line_y_tol
    lines_by_y = []
    curr_line, curr_y = [], None
    for w in words_sorted:
        w_y = w["top"]
        w_fs = w.get("size", 12)
        if not curr_line:
            curr_line = [w]
            curr_y = w_y
        elif abs(w_y - curr_y) < line_y_tol(w_fs):
            curr_line.append(w)
            curr_y = (curr_y + w_y) / 2  # running average, robust to small drift
        else:
            lines_by_y.append(curr_line)
            curr_line = [w]
            curr_y = w_y
    if curr_line:
        lines_by_y.append(curr_line)

    # Step 2: for each grouped line, join words by x-order, preserving y-line order!
    lines = []
    for line_words in lines_by_y:
        line_words_sorted = sorted(line_words, key=lambda w: w["x0"])
        combined_text = ' '.join([w["text"].strip() for w in line_words_sorted if w["text"].strip()])
        # Only skip if the line is empty or just a section number
        if not combined_text.strip() or re.fullmatch(r'\d+[.)]?', combined_text.strip()):
            continue
        # Compute dominant font, etc.
        font_sizes = [w.get("size", 12) for w in line_words]
        font_names = [w.get("fontname", "") for w in line_words]
        uprights = [w.get("upright", True) for w in line_words]
        dominant_font_size = max(set(font_sizes), key=font_sizes.count)
        dominant_font_name = max(set(font_names), key=font_names.count) if font_names else ""
        is_upright = sum(uprights) > len(uprights) / 2
        is_bold = detect_bold_text(dominant_font_name, combined_text)
        min_x0, max_x1 = min(w["x0"] for w in line_words), max(w["x1"] for w in line_words)
        min_top, max_bot = min(w["top"] for w in line_words), max(w["bottom"] for w in line_words)

        line_dict = dict(
            text=combined_text,
            fontname=dominant_font_name,
            size=dominant_font_size,
            upright=is_upright,
            is_bold=is_bold,
            x0=min_x0, x1=max_x1, top=np.median([w["top"] for w in line_words]), bottom=max_bot,
            page_width=page_width, page_height=page_height,
            percent_alpha=len([c for c in combined_text if c.isalpha()]) / max(1, len(combined_text)),
            fraction_uppercase=len([c for c in combined_text if c.isupper()]) / max(1, len(combined_text)),
            ends_with_colon=int(combined_text.strip().endswith(':')),
            is_very_short_heading_candidate=int(len(combined_text.split()) == 1 and len(combined_text) < 8 and combined_text.strip().endswith(':'))
        )
        lines.append(line_dict)
    return lines


def create_line_from_words(line_words: List[Dict], page_width: float, page_height: float) -> Dict:
    if not line_words: return None
    combined_text = ' '.join(w["text"].strip() for w in line_words if w["text"].strip())
    combined_text = ' '.join(combined_text.split())
    combined_text = collapse_exactly4_repeats(combined_text)
    font_sizes = [w.get("size", 12) for w in line_words]
    font_names = [w.get("fontname", "") for w in line_words]
    uprights = [w.get("upright", True) for w in line_words]
    percent_alpha = len([c for c in combined_text if c.isalpha()]) / max(1, len(combined_text))
    fraction_upper = len([c for c in combined_text if c.isupper()]) / max(1, len(combined_text))
    ends_with_colon = int(combined_text.strip().endswith(':'))
    is_very_short_candidate = int(len(combined_text.split()) == 1 and len(combined_text) < 8 and combined_text.strip().endswith(':'))
    dominant_font_size = max(set(font_sizes), key=font_sizes.count)
    dominant_font_name = max(set(font_names), key=font_names.count) if font_names else ""
    is_upright = sum(uprights) > len(uprights) / 2
    is_bold = detect_bold_text(dominant_font_name, combined_text)
    min_x0, max_x1 = min(w["x0"] for w in line_words), max(w["x1"] for w in line_words)
    min_top, max_bot = min(w["top"] for w in line_words), max(w["bottom"] for w in line_words)
    return {
        "text": combined_text,
        "fontname": dominant_font_name,
        "size": dominant_font_size,
        "upright": is_upright,
        "is_bold": is_bold,
        "x0": min_x0, "x1": max_x1, "top": min_top, "bottom": max_bot,
        "page_width": page_width, "page_height": page_height,
        "percent_alpha": percent_alpha,
        "fraction_uppercase": fraction_upper,
        "ends_with_colon": ends_with_colon,
        "is_very_short_heading_candidate": is_very_short_candidate,
    }

def preprocess_pdf_enhanced(pdf_path: str) -> List[Dict]:
    all_page_data = []
    with pdfplumber.open(pdf_path) as pdf:
        n_pages = len(pdf.pages)
        temp_page_data = []
        for i, page in enumerate(pdf.pages):
            words = page.extract_words(extra_attrs=["fontname", "size", "upright"])
            words = exclude_words_in_tables(words, page)
            page_height = float(page.height)
            page_width = float(page.width)
            lines = group_words_into_lines(words, page_width, page_height, n_pages)
            temp_page_data.append({
                "page": i + 1,
                "lines": lines,
                "page_width": page_width,
                "page_height": page_height,
            })
        header_footer_patterns = detect_frequency_based_headers_footers(temp_page_data, n_pages)
        for i, page in enumerate(pdf.pages):
            words = page.extract_words(extra_attrs=["fontname", "size", "upright"])
            words = exclude_words_in_tables(words, page)
            page_height = float(page.height)
            page_width = float(page.width)
            lines = group_words_into_lines(words, page_width, page_height, n_pages, header_footer_patterns)
            all_page_data.append({
                "page": i + 1,
                "lines": lines,
                "page_width": page_width,
                "page_height": page_height,
            })
    return all_page_data

def extract_enhanced_features_from_preprocessed_data(page_data: List[Dict]) -> List[Dict]:
    rows = []
    all_font_sizes = []
    for page in page_data:
        for line in page["lines"]:
            all_font_sizes.append(line.get("size", 12))
    if not all_font_sizes:
        return []
    avg, mx, std = np.mean(all_font_sizes), max(all_font_sizes), np.std(all_font_sizes)
    for page in page_data:
        page_num = page["page"]
        lines, pw, ph = page["lines"], page["page_width"], page["page_height"]
        if not lines:
            continue
        for line in lines:
            txt, fs = line["text"], line.get("size", 12)
            rows.append({
                "text": txt,
                "page_num": page_num,
                "font_size": float(fs),
                "font_size_relative": fs / avg if avg else 1.0,
                "font_size_zscore": (fs - avg) / std if std else 0,
                "is_largest_font": int(fs >= mx * 0.95),
                "font_name": line.get("fontname", ""),
                "is_all_caps": int(txt.isupper()),
                "is_title_case": int(txt.istitle()),
                "starts_with_number": int(bool(txt and (txt[0].isdigit() or re.match(r'^\d+(\.\d+)*', txt)))),
                "text_length": len(txt),
                "word_count": len(txt.split()),
                "has_colon": int(':' in txt),
                "is_bold": int(line.get("is_bold", False)),
                "is_upright": int(line.get("upright", True)),
                "x0": line["x0"],
                "x1": line["x1"],
                "y0": line["top"],
                "y1": line["bottom"],
                "width": line["x1"] - line["x0"],
                "height": line["bottom"] - line["top"],
                "x_relative": line["x0"] / pw if pw else 0,
                "y_relative": line["top"] / ph if ph else 0,
                "is_left_aligned": int(line["x0"] < pw * 0.15),
                "is_centered": int(abs(line["x0"] + (line["x1"] - line["x0"]) / 2 - pw / 2) < pw * 0.1),
                "is_top_area": int(line["top"] < ph * 0.2),
                "contains_heading_keywords": int(any(kw in txt.lower() for kw in [
                    'chapter', 'section', 'introduction', 'conclusion', 'summary', 'abstract', 'background',
                    'method', 'result', 'discussion', 'appendix', 'reference', 'overview', 'contents',
                    'table of contents', 'acknowledgement', 'revision history'
                ])),
                "has_section_number": int(bool(re.match(r'^\d+(\.\d+)*', txt.strip()))),
                "looks_like_heading": int(
                    len(txt.split()) <= 15 and not txt.endswith('.') and
                    (txt.isupper() or txt.istitle() or line.get("is_bold", False) or
                    any(kw in txt.lower() for kw in ['chapter', 'section', 'appendix', 'table', 'figure']))
                ),
                "is_dot_leader": int(is_dot_leader(txt)),
                "percent_alpha": line.get("percent_alpha", 0.0),
                "fraction_uppercase": line.get("fraction_uppercase", 0.0),
                "ends_with_colon": line.get("ends_with_colon", 0),
                "is_very_short_heading_candidate": line.get("is_very_short_heading_candidate", 0),
                "page_width": pw,
                "page_height": ph,
            })
    return rows
