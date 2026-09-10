#!/usr/bin/env python3
"""
Gate 4: BM25 comparison between niyah C index and rank_bm25 reference.
Also measures add_document scaling to find fitted complexity exponent.
"""

import subprocess, sys, os, math, time, json, ctypes, tempfile, struct

# --- Build a tiny C harness to exercise the search index via the shared lib ---
HARNESS_SRC = r"""
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../search/niyah_index.h"

int main(int argc, char** argv) {
    if (argc < 3) { fprintf(stderr, "usage: harness doc_file query\n"); return 1; }

    FILE* f = fopen(argv[1], "r");
    if (!f) { perror("fopen"); return 1; }

    NiyahInvertedIndex idx;
    niyah_index_init(&idx, 1.2, 0.75);

    char line[1<<20];
    uint32_t doc_id = 0;
    while (fgets(line, sizeof(line), f)) {
        size_t n = strlen(line);
        if (n > 0 && line[n-1] == '\n') line[n-1] = '\0';
        if (line[0] == '\0') continue;
        NiyahDocument doc;
        doc.document_id = ++doc_id;
        doc.text = line;
        niyah_index_add_document(&idx, &doc);
    }
    fclose(f);

    NiyahSearchHit hits[64];
    size_t count = niyah_index_search(&idx, argv[2], hits, 64);
    for (size_t i = 0; i < count; ++i) {
        printf("doc=%u score=%.10f\n", hits[i].document_id, hits[i].score);
    }

    // Emit average_document_length and doc count
    printf("avg_len=%.6f ndocs=%zu\n", idx.average_document_length, idx.document_count);

    // Print each document's stored term_count
    for (size_t i = 0; i < idx.document_count; ++i) {
        printf("stored_term_count[%zu]=%u\n", i, idx.documents[i].term_count);
    }

    niyah_index_free(&idx);
    return 0;
}
"""

import os, subprocess, tempfile, re

REPO = "/workspace/Niyah.Engine"
BUILD = os.path.join(REPO, "build/native")

# Write harness
harness_src = os.path.join(REPO, "bm25_harness.c")
harness_bin = os.path.join(REPO, "bm25_harness")

with open(harness_src, "w") as f:
    f.write(HARNESS_SRC)

# Compile harness against the niyah static library
r = subprocess.run([
    "gcc", "-O2", "-I", os.path.join(REPO, "search"),
    "-I", os.path.join(REPO, "native"),
    harness_src,
    os.path.join(REPO, "search/niyah_index.c"),
    "-o", harness_bin, "-lm"
], capture_output=True, text=True)
if r.returncode != 0:
    print("COMPILE ERROR:", r.stderr)
    sys.exit(1)
print(f"harness compiled: rc={r.returncode}")

# --- Generate documents of target lengths ---
import re as _re

def make_doc(n_tokens):
    """Generate a document with exactly n_tokens distinct lowercase words."""
    words = []
    for i in range(n_tokens):
        # Word format: w<base36> ensures diversity, each is a valid token
        words.append(f"w{i:04x}")
    return " ".join(words)

DOC_SIZES = [100, 1000, 1024, 1025, 5000]
QUERY = "w0001 w0002 w0003"  # tokens that appear in all docs

doc_file = os.path.join(REPO, "bm25_docs.txt")
with open(doc_file, "w") as f:
    for n in DOC_SIZES:
        f.write(make_doc(n) + "\n")

print(f"wrote {len(DOC_SIZES)} docs: {DOC_SIZES}")

# Run C harness
r = subprocess.run([harness_bin, doc_file, QUERY], capture_output=True, text=True)
print("C harness rc:", r.returncode)
print(r.stdout)
c_output = r.stdout

# Parse C scores
c_scores = {}
for m in re.finditer(r"doc=(\d+) score=([\d.]+)", c_output):
    c_scores[int(m.group(1))] = float(m.group(2))

stored_term_counts = {}
for m in re.finditer(r"stored_term_count\[(\d+)\]=(\d+)", c_output):
    stored_term_counts[int(m.group(1))] = int(m.group(2))

print("Stored term counts:", stored_term_counts)
print("C scores:", c_scores)

# --- Python BM25 reference using rank_bm25 ---
from rank_bm25 import BM25Okapi

# Tokenizer that matches niyah's: split on non-(alnum|_|-), lowercase
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

# Truncate at NIYAH_DOCUMENT_TOKEN_LIMIT=1024 AS THE C CODE DOES
LIMIT = 1024
docs_raw = [make_doc(n) for n in DOC_SIZES]
corpus_truncated = [niyah_tokenize(d)[:LIMIT] for d in docs_raw]
corpus_full = [niyah_tokenize(d) for d in docs_raw]

# rank_bm25 with TRUNCATED corpus (mirrors C behavior)
bm25_trunc = BM25Okapi(corpus_truncated, k1=1.2, b=0.75)
query_tokens = niyah_tokenize(QUERY)
scores_trunc = bm25_trunc.get_scores(query_tokens)

# rank_bm25 with FULL corpus (what it should be without truncation bug)
bm25_full = BM25Okapi(corpus_full, k1=1.2, b=0.75)
scores_full = bm25_full.get_scores(query_tokens)

print("\n=== BM25 COMPARISON (C vs Python-truncated vs Python-full) ===")
print(f"{'Doc':>5} {'Size':>6} {'Stored':>8} {'C score':>14} {'Py-trunc':>14} {'Py-full':>14} {'|C-Pytrunc|':>14}")
max_abs_delta_trunc = 0.0
max_abs_delta_full = 0.0
rank_inversions = 0
first_diverge_len = None

for i, (n, st) in enumerate(zip(DOC_SIZES, [stored_term_counts.get(i, 0) for i in range(len(DOC_SIZES))])):
    doc_id = i + 1
    cs = c_scores.get(doc_id, 0.0)
    pt = scores_trunc[i]
    pf = scores_full[i]
    delta = abs(cs - pt)
    if delta > max_abs_delta_trunc:
        max_abs_delta_trunc = delta
    delta_full = abs(cs - pf)
    if delta_full > max_abs_delta_full:
        max_abs_delta_full = delta_full
    print(f"{doc_id:>5} {n:>6} {st:>8} {cs:>14.8f} {pt:>14.8f} {pf:>14.8f} {delta:>14.2e}")

# Rank inversion check (full vs truncated)
ranks_c = sorted(range(len(DOC_SIZES)), key=lambda i: c_scores.get(i+1, 0.0), reverse=True)
ranks_full = sorted(range(len(DOC_SIZES)), key=lambda i: scores_full[i], reverse=True)
for pos, (rc, rf) in enumerate(zip(ranks_c, ranks_full)):
    if rc != rf:
        rank_inversions += 1
        if first_diverge_len is None:
            first_diverge_len = DOC_SIZES[rc]

print(f"\nMAX_ABS_SCORE_DELTA (C vs Python-truncated): {max_abs_delta_trunc:.2e}")
print(f"MAX_ABS_SCORE_DELTA (C vs Python-full):      {max_abs_delta_full:.2e}")
print(f"RANK_INVERSIONS (C vs Python-full):           {rank_inversions}")
print(f"FIRST_DOC_LENGTH_WHERE_RANKING_DIVERGES:      {first_diverge_len}")

# Confirm truncation effect: docs with >1024 tokens should have stored_term_count=1024
print("\n=== TRUNCATION VERIFICATION ===")
for i, n in enumerate(DOC_SIZES):
    st = stored_term_counts.get(i, 0)
    expected = min(n, LIMIT)
    match = "OK" if st == expected else f"MISMATCH(expected {expected})"
    print(f"  doc {i+1}: true_tokens={n}, stored_term_count={st} [{match}]")

# --- Complexity measurement ---
print("\n=== COMPLEXITY MEASUREMENT: add_document scaling ===")

SCALING_SIZES = [10, 100, 1000]

# Write and time harness
TIMING_SRC = r"""
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "../search/niyah_index.h"

int main(int argc, char** argv) {
    if (argc < 2) return 1;
    int n = atoi(argv[1]);
    
    NiyahInvertedIndex idx;
    niyah_index_init(&idx, 1.2, 0.75);
    
    char text[64];
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int i = 0; i < n; i++) {
        snprintf(text, sizeof(text), "word%d unique%d common", i, i);
        NiyahDocument doc;
        doc.document_id = (uint32_t)(i + 1);
        doc.text = text;
        niyah_index_add_document(&idx, &doc);
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);
    
    double elapsed = (t1.tv_sec - t0.tv_sec)*1e9 + (t1.tv_nsec - t0.tv_nsec);
    printf("n=%d elapsed_ns=%.0f\n", n, elapsed);
    
    niyah_index_free(&idx);
    return 0;
}
"""

timing_src = os.path.join(REPO, "bm25_timing.c")
timing_bin = os.path.join(REPO, "bm25_timing")
with open(timing_src, "w") as f:
    f.write(TIMING_SRC)

r = subprocess.run([
    "gcc", "-O2", "-I", os.path.join(REPO, "search"),
    "-I", os.path.join(REPO, "native"),
    timing_src,
    os.path.join(REPO, "search/niyah_index.c"),
    "-o", timing_bin, "-lm"
], capture_output=True, text=True)
if r.returncode != 0:
    print("TIMING COMPILE ERROR:", r.stderr)
else:
    times = {}
    for n in SCALING_SIZES:
        # run 3 times, take median
        runs = []
        for _ in range(3):
            r2 = subprocess.run([timing_bin, str(n)], capture_output=True, text=True)
            m = re.search(r"elapsed_ns=([\d.]+)", r2.stdout)
            if m:
                runs.append(float(m.group(1)))
        if runs:
            times[n] = sorted(runs)[1]  # median
            print(f"  add_document x{n}: median={times[n]/1e6:.3f} ms")
    
    if len(times) >= 2:
        ns = sorted(times.keys())
        # Fit exponent: t = C * n^alpha => alpha = log(t2/t1) / log(n2/n1)
        exponents = []
        for i in range(len(ns)-1):
            a = math.log(times[ns[i+1]] / times[ns[i]])
            b = math.log(ns[i+1] / ns[i])
            if b > 0:
                exponents.append(a/b)
        avg_exp = sum(exponents)/len(exponents)
        print(f"  FITTED_EXPONENT={avg_exp:.3f}")
        if avg_exp < 1.3:
            print("  MEASURED_COMPLEXITY: ~O(n) — consistent with linear claim")
        elif avg_exp < 2.1:
            print(f"  MEASURED_COMPLEXITY: ~O(n^{avg_exp:.2f}) — super-linear but sub-quadratic")
        else:
            print(f"  MEASURED_COMPLEXITY: ~O(n^{avg_exp:.2f}) — super-linear, contradicts linear claim")
