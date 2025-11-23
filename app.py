import streamlit as st
import pandas as pd
import re, math, json, os
from collections import Counter
from typing import List, Dict
EXCEL_PATH = "/mnt/data/Case study for interns.xlsx"  
USE_SENT_TRANSFORMERS = True
MODEL_NAME = "all-MiniLM-L6-v2"  
def safe_read_excel(path):
    try:
        df = pd.read_excel(path, sheet_name="Rubrics", header=None)
        return df
    except Exception as e:
        st.error(f"Could not open rubric Excel at {path}: {e}")
        return None

def extract_rubric_from_df(df):
    """
    Heuristic extraction: returns list of criteria dicts:
    [{ 'name':..., 'description':..., 'weight':int, 'keywords': [..], 'min_words':int or None, 'max_words':int or None }]
    """
    text_lines = []
    for i in range(len(df)):
        joined = " ".join(df.iloc[i].astype(str).tolist()).strip()
        if joined and joined != 'nan':
            text_lines.append((i, joined))
    criteria = []
    cand_names = ["Content & Structure","Communication Skills","Language & Grammar","Confidence","Pronunciation","Fluency","Length","Time Management"]
    for name in cand_names:
        for i, line in text_lines:
            if name.lower() in line.lower():
                nearby = " ".join([l for idx,l in text_lines if idx >= i and idx < i+8])
                nums = re.findall(r'\b\d+\b', nearby)
                weight = int(nums[0]) if nums else 10
                kw_match = re.search(r'Keywords[:\-]*\s*([A-Za-z0-9, ]+)', nearby, re.I)
                keywords = []
                if kw_match:
                    keywords = [k.strip() for k in kw_match.group(1).split(",") if k.strip()]
                mn = None; mx = None
                mm = re.search(r'(min|max)\s*[:=]?\s*(\d+)', nearby, re.I)
                criteria.append({"name": name, "description": line, "weight": weight, "keywords": keywords, "min_words": mn, "max_words": mx})
                break
    if not criteria:
        for i, line in text_lines:
            if re.search(r'Content|Structure|Language|Confidence|Pronunciation|Fluency', line, re.I):
                nums = re.findall(r'\b\d+\b', line)
                weight = int(nums[-1]) if nums else 10
                criteria.append({"name": line.split()[0], "description": line, "weight": weight, "keywords": [], "min_words": None, "max_words": None})
    total = sum(c['weight'] for c in criteria) if criteria else 0
    if total == 0:
        criteria = [
            {"name":"Content & Structure","description":"", "weight":30,"keywords":[],"min_words":60,"max_words":160},
            {"name":"Communication Skills","description":"", "weight":25,"keywords":[],"min_words":None,"max_words":None},
            {"name":"Language & Grammar","description":"", "weight":20,"keywords":[],"min_words":None,"max_words":None},
            {"name":"Confidence","description":"", "weight":15,"keywords":[],"min_words":None,"max_words":None},
            {"name":"Pronunciation/Fluency","description":"", "weight":10,"keywords":[],"min_words":None,"max_words":None},
        ]
    else:
        scale = 100.0 / total
        for c in criteria:
            c['weight'] = round(c['weight'] * scale, 2)
    return criteria

FILLERS = {"um","uh","like","you know","erm","hmm","ah","okay","ok","so","actually","basically"}
def word_tokens(text):
    return [w for w in re.findall(r"\b\w+'?\w*\b", text)]

def count_fillers(text):
    words = [w.lower() for w in word_tokens(text)]
    return sum(1 for w in words if w in FILLERS)

def type_token_ratio(text):
    words = [w.lower() for w in word_tokens(text) if w.isalpha()]
    if not words:
        return 0.0
    return len(set(words)) / len(words)

def length_score(word_count, ideal_low=60, ideal_high=140):
    if ideal_low <= word_count <= ideal_high:
        return 100.0
    if word_count < ideal_low:
        return max(0.0, 100.0 * (word_count / ideal_low))
    else:
        return max(0.0, 100.0 * (1 - (word_count - ideal_high) / (word_count + 1)))

def intelligibility_score(text):
    inaudible = len(re.findall(r'\[inaudible\]|\[unclear\]|\[.*?unclear.*?\]', text, flags=re.I))
    return max(0.0, 100.0 - inaudible*30.0)

semantic_model = None
def load_semantic_model():
    global semantic_model
    if not USE_SENT_TRANSFORMERS:
        return None
    try:
        from sentence_transformers import SentenceTransformer, util
        semantic_model = SentenceTransformer(MODEL_NAME)
        return semantic_model
    except Exception as e:
        st.warning(f"sentence-transformers not usable ({e}). Falling back to heuristics.")
        return None

def semantic_similarity(a: str, b: str):
    if semantic_model:
        from sentence_transformers import util
        emb_a = semantic_model.encode(a, convert_to_tensor=True)
        emb_b = semantic_model.encode(b, convert_to_tensor=True)
        return float(util.cos_sim(emb_a, emb_b).item())
    sa = set(word_tokens(a.lower()))
    sb = set(word_tokens(b.lower()))
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)

def score_criterion(transcript: str, criterion: dict):
    """
    Returns {'score': float (0-100), 'evidence': dict}
    """
    name = criterion['name']
    weight = criterion['weight']
    desc = criterion.get('description','')
    kws = criterion.get('keywords') or []
    wc = len(word_tokens(transcript))
    evidence = {}
    if kws:
        lower = transcript.lower()
        found = [k for k in kws if k.lower() in lower]
        kw_score = min(100.0, 100.0 * len(found) / len(kws))
        evidence['keywords_found'] = found
        evidence['kw_score'] = kw_score
    else:
        kw_score = None
    sem = semantic_similarity(transcript, desc) if desc else 0.0
    sem_score = min(1.0, max(0.0, sem)) * 100.0
    evidence['semantic_similarity'] = round(sem_score, 3)
    if "length" in name.lower() or "time" in name.lower():
        score = length_score(wc, ideal_low=criterion.get('min_words') or 60, ideal_high=criterion.get('max_words') or 140)
    elif "language" in name.lower() or "grammar" in name.lower():
        
        ttr = type_token_ratio(transcript)
        filler_rate = count_fillers(transcript) / max(1, wc)
        score = max(0.0, min(100.0, (ttr*100.0)*0.7 + (1 - filler_rate)*100.0*0.3))
        evidence['ttr'] = round(ttr,3); evidence['filler_rate'] = round(filler_rate,3)
    elif "confidence" in name.lower():
        filler_rate = count_fillers(transcript) / max(1, wc)
        punct = transcript.count('!') + transcript.count('.')*0.1
        score = max(0.0, min(100.0, (1 - filler_rate)*70 + min(30, punct*10)))
        evidence['filler_rate'] = round(filler_rate,3)
    elif "content" in name.lower() or "structure" in name.lower():
        
        if kw_score is None:
            score = sem_score * 0.8 + (length_score(wc) * 0.2)
        else:
            score = (kw_score*0.5 + sem_score*0.4 + length_score(wc)*0.1)
    else:
      
        score = 0.6*sem_score + 0.4*length_score(wc)
    
    if kw_score is not None:
        
        score = kw_score*0.4 + sem_score*0.4 + length_score(wc)*0.2
    evidence['computed_raw'] = round(score,2)
    return max(0.0, min(100.0, round(score,2))), evidence

def overall_scoring(transcript: str, rubric_list: List[dict]):
    per = []
    total_weight = sum(r['weight'] for r in rubric_list) or 100.0
    weighted_sum = 0.0
    for c in rubric_list:
        sc, ev = score_criterion(transcript, c)
        per.append({"name": c['name'], "weight": c['weight'], "score": sc, "evidence": ev})
        weighted_sum += sc * c['weight']
    final = round(weighted_sum / total_weight, 2) if total_weight else round(sum(p['score'] for p in per)/len(per),2)
    return final, per


st.set_page_config(page_title="Nirmaan — Intro Evaluator", layout="wide")
st.markdown("<h1 style='text-align:center'>Nirmaan — Spoken Introduction Evaluator</h1>", unsafe_allow_html=True)
st.markdown("**Professional UI** • Reads rubric from the provided Excel and combines rule-based + semantic scoring.")

col1, col2 = st.columns([2,1])
with col1:
    st.subheader("Input transcript")
    txt = st.text_area("Paste transcript here", height=300)
    uploaded = st.file_uploader("OR upload a .txt file", type=['txt'])
    if uploaded and not txt:
        txt = uploaded.read().decode('utf-8')
    st.markdown("**Optional:** provide comma-separated keywords to boost Content scoring (overrides rubric keywords).")
    kw_line = st.text_input("Extra keywords (comma-separated)")
    extra_keywords = [k.strip() for k in kw_line.split(",")] if kw_line else None

with col2:
    st.subheader("Rubric (extracted)")
    df = safe_read_excel(EXCEL_PATH)
    if df is not None:
        rubric_list = extract_rubric_from_df(df)
       
        rub_tbl = pd.DataFrame([{"Criterion":r['name'], "Weight":r['weight'], "Keywords":", ".join(r.get('keywords') or [])} for r in rubric_list])
        st.dataframe(rub_tbl, height=300)
    else:
        rubric_list = []

st.markdown("---")
if st.button("Load semantic model (optional)"):
    load_semantic_model()
    if semantic_model:
        st.success("Semantic model loaded. Semantic similarity will be used.")
    else:
        st.info("Semantic model not available — app will use heuristics.")

if st.button("Score transcript"):
    if not txt or txt.strip()=="":
        st.error("Please paste or upload a transcript first.")
    else:
        if extra_keywords:
            
            found = False
            for r in rubric_list:
                if "content" in r['name'].lower():
                    r['keywords'] = extra_keywords
                    found = True
            if not found:
                rubric_list.insert(0, {"name":"Content & Structure","description":"", "weight":30,"keywords":extra_keywords,"min_words":60,"max_words":140})
        with st.spinner("Scoring..."):
            final, per = overall_scoring(txt, rubric_list)
        # Results panel
        st.markdown("### Results")
        c1, c2 = st.columns([1,2])
        with c1:
            st.metric("Overall score", f"{final} / 100")
            st.write(f"Word count: {len(word_tokens(txt))}")
        with c2:
            for p in per:
                st.progress(int(p['score']))
                st.markdown(f"**{p['name']}** — weight: {p['weight']} — score: {p['score']}")
                ev = p['evidence']
                show = []
                if 'keywords_found' in ev:
                    show.append("Keywords found: " + (", ".join(ev['keywords_found']) or "None"))
                if 'semantic_similarity' in ev:
                    show.append(f"Semantic sim: {ev['semantic_similarity']}")
                if 'ttr' in ev:
                    show.append(f"TTR: {ev['ttr']}")
                if 'filler_rate' in ev:
                    show.append(f"Filler rate: {ev['filler_rate']}")
                st.caption(" · ".join(show))
        st.markdown("#### Per-criterion JSON (copy/paste for submission)")
        st.json({"overall": final, "per_criterion": per, "words": len(word_tokens(txt))})
        st.balloons()
