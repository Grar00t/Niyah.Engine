#!/usr/bin/env python3
"""
Gate 4: BM25 comparison using identical IDF formula as niyah C code.
niyah uses: idf = log1p((N - df + 0.5) / (df + 0.5))
rank_bm25 uses: idf = log((N - df + 0.5) / (df + 0.5))  [can be negative]
This script uses the niyah formula for a fair comparison.
"""
import subprocess, re, math, os

REPO = "/workspace/Niyah.Engine"
LIMIT = 1024

def niyah_tokenize(text):
    tokens = []
    buf = ""
    for ch in text:
        o = ord(ch)
        if ch.isalnum() or ch == '_' or ch == '-' or o >= 0x80:
            buf += ch.lower() if o < 0x80 else ch
        else:
            if buf:
                tokens.append(buf)
                buf = ""
    if buf:
        tokens.append(buf)
    return tokens

def make_doc(n_tokens):
    return " ".join(f"w{i:04x}" for i in range(n_tokens))

DOC_SIZES = [100, 1000, 1024, 1025, 5000]
QUERY = "w0001 w0002 w0003"

docs_raw = [make_doc(n) for n in DOC_SIZES]
query_tokens = niyah_tokenize(QUERY)

# Python BM25 with TRUNCATED corpus (mirrors C)
def bm25_score_niyah(k1, b, n_docs, avg_len, doc_len, df, tf):
    """niyah BM25 formula: idf = log1p((N - df + 0.5) / (df + 0.5))"""
    if tf == 0 or doc_len == 0 or n_docs == 0:
        return 0.0
    idf = math.log1p((n_docs - df + 0.5) / (df + 0.5))
    norm = k1 * (1.0 - b + b * (doc_len / avg_len))
    score_denom = tf + norm
    if score_denom <= 0:
        return 0.0
    return idf * (tf * (k1 + 1.0) / score_denom)

def score_corpus(tokenized_docs, query_toks, truncate=None):
    """Score query against corpus. truncate=N means cap each doc at N tokens."""
    k1, b = 1.2, 0.75
    proc = [t[:truncate] if truncate else t for t in tokenized_docs]
    lengths = [len(d) for d in proc]
    avg_len = sum(lengths) / len(lengths) if lengths else 1.0
    n_docs = len(proc)
    
    # Build inverted index
    from collections import Counter, defaultdict
    inv = defaultdict(list)  # term -> [(doc_idx, tf)]
    df = {}
    for i, doc in enumerate(proc):
        c = Counter(doc)
        for term, freq in c.items():
            inv[term].append((i, freq))
            df[term] = df.get(term, 0) + 1
    
    scores = [0.0] * n_docs
    seen = set()
    for qt in query_toks:
        if qt in seen:
            continue
        seen.add(qt)
        if qt not in inv:
            continue
        dfi = df[qt]
        for (di, tf) in inv[qt]:
            scores[di] += bm25_score_niyah(k1, b, n_docs, avg_len, lengths[di], dfi, tf)
    return scores

tokenized_full = [niyah_tokenize(d) for d in docs_raw]

scores_trunc = score_corpus(tokenized_full, query_tokens, truncate=LIMIT)
scores_full  = score_corpus(tokenized_full, query_tokens, truncate=None)

# Get C scores
r = subprocess.run([os.path.join(REPO, "bm25_harness"), 
                    os.path.join(REPO, "bm25_docs.txt"), QUERY],
                   capture_output=True, text=True)
c_scores_raw = {}
stored_tc = {}
for m in re.finditer(r"doc=(\d+) score=([\d.]+)", r.stdout):
    c_scores_raw[int(m.group(1))] = float(m.group(2))
for m in re.finditer(r"stored_term_count\[(\d+)\]=(\d+)", r.stdout):
    stored_tc[int(m.group(1))] = int(m.group(2))

print("=== GATE 4: BM25 CROSS-VALIDATION (identical IDF formula) ===")
print(f"IDF formula: log1p((N - df + 0.5) / (df + 0.5))  [niyah formula]")
print(f"k1=1.2, b=0.75")
print()
print(f"{'Doc':>5} {'Tokens':>7} {'Stored':>7} {'C_score':>14} {'Py_trunc':>14} {'|C-Py|':>12} {'Py_full':>14}")

max_abs_delta = 0.0
for i, n in enumerate(DOC_SIZES):
    did = i + 1
    cs = c_scores_raw.get(did, 0.0)
    pt = scores_trunc[i]
    pf = scores_full[i]
    delta = abs(cs - pt)
    if delta > max_abs_delta:
        max_abs_delta = delta
    print(f"{did:>5} {n:>7} {stored_tc.get(i,0):>7} {cs:>14.8f} {pt:>14.8f} {delta:>12.2e} {pf:>14.8f}")

# Rank inversions: C (truncated) vs full Python
ranks_c   = sorted(range(len(DOC_SIZES)), key=lambda i: c_scores_raw.get(i+1, 0.0), reverse=True)
ranks_full = sorted(range(len(DOC_SIZES)), key=lambda i: scores_full[i], reverse=True)

inversions = 0
first_div = None
for pos, (rc, rf) in enumerate(zip(ranks_c, ranks_full)):
    if rc != rf:
        inversions += 1
        if first_div is None:
            first_div = DOC_SIZES[rc]

print()
print(f"DOCS=5  QUERIES=1")
print(f"MAX_ABS_SCORE_DELTA={max_abs_delta:.3e}")
print(f"RANK_INVERSIONS={inversions}")
print(f"FIRST_DOC_LENGTH_WHERE_RANKING_DIVERGES={first_div}")

print()
print("=== TRUNCATION EFFECT DETAIL ===")
print(f"docs 4 (1025 tokens) and 5 (5000 tokens) both truncated to stored_term_count=1024")
print(f"C scores for docs 3,4,5: {c_scores_raw.get(3):.8f}, {c_scores_raw.get(4):.8f}, {c_scores_raw.get(5):.8f}")
print(f"Are they identical? {c_scores_raw.get(3)==c_scores_raw.get(4)==c_scores_raw.get(5)}")
print(f"Python-full scores for docs 3,4,5: {scores_full[2]:.8f}, {scores_full[3]:.8f}, {scores_full[4]:.8f}")
print(f"Python-full distinguishes 1024 vs 1025 vs 5000 tokens: {len(set([round(scores_full[i],6) for i in [2,3,4]])) > 1}")
