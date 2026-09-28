import os
import sys
import time
import pickle
import csv
from collections import Counter
import numpy as np

BASE_DIR = r"C:\Users\palfi\Downloads\6ab10eb3b23ba_student_resource\student_resource\dataset"
sys.path.insert(0, BASE_DIR)

from src.normalization import normalize_country, normalize_text, standardize_address_text
from src.tokenization import extract_name_tokens, extract_address_tokens, extract_name_ngrams
from src.features import FEATURE_NAMES, extract_pair_features

print("Testing high-speed partitioned candidate ranker and matcher...")
with open(os.path.join(BASE_DIR, "models", "er_matcher_final_96_19.pkl"), "rb") as f:
    artifact = pickle.load(f)
model = artifact["model"]

# Index France targets
idx_postings = {}
target_records = {}

t0 = time.time()
for p in ["test/test_source2.tsv", "test/test_source3.tsv"]:
    fpath = os.path.join(BASE_DIR, p)
    with open(fpath, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader)
        pos = {name: i for i, name in enumerate(header)}
        for row in reader:
            if normalize_country(row[pos["country"]]) == "FRANCE":
                eid = row[pos["entity_id"]]
                name = normalize_text(row[pos["business_name"]])
                addr = standardize_address_text(row[pos["business_address"]])
                target_records[eid] = (name, addr, "FRANCE")
                
                # Add to postings
                toks = set(extract_name_tokens(name))
                toks.update(extract_address_tokens(addr))
                toks.update(extract_name_ngrams(name, n=4))
                for t in toks:
                    if t not in idx_postings:
                        idx_postings[t] = []
                    idx_postings[t].append(eid)

print(f"Indexed {len(target_records):,} France targets ({len(idx_postings):,} unique tokens) in {time.time()-t0:.2f}s")

# Test 10,000 queries
s1_sample = []
s1_path = os.path.join(BASE_DIR, "test/test_source1.tsv")
with open(s1_path, "r", encoding="utf-8") as f:
    reader = csv.reader(f, delimiter="\t")
    header = next(reader)
    pos = {name: i for i, name in enumerate(header)}
    for row in reader:
        if normalize_country(row[pos["country"]]) == "FRANCE":
            eid = row[pos["entity_id"]]
            name = normalize_text(row[pos["business_name"]])
            addr = standardize_address_text(row[pos["business_address"]])
            s1_sample.append((eid, name, addr, "FRANCE"))
            if len(s1_sample) >= 10000:
                break

print(f"Scoring 10,000 France queries...")
t1 = time.time()

feature_rows = []
offsets = []
cands_count = 0

for eid, name, addr, cntry in s1_sample:
    n_toks = extract_name_tokens(name)
    a_toks = extract_address_tokens(addr)
    ng_toks = extract_name_ngrams(name, n=4)
    
    counts = Counter()
    for t in n_toks:
        posts = idx_postings.get(t)
        if posts and len(posts) <= 2000:
            for cid in posts:
                counts[cid] += 2
    for t in a_toks:
        posts = idx_postings.get(t)
        if posts:
            cap = 4000 if (t.startswith("sh_") or t.startswith("num_")) else 1500
            if len(posts) <= cap:
                for cid in posts:
                    counts[cid] += 1
    for t in ng_toks:
        posts = idx_postings.get(t)
        if posts and len(posts) <= 200:
            for cid in posts:
                counts[cid] += 1
                
    if not counts:
        all_t = n_toks + a_toks + ng_toks
        avail = [(len(idx_postings[t]), t) for t in all_t if t in idx_postings]
        if avail:
            rarest = min(avail)[1]
            for cid in idx_postings[rarest][:50]:
                counts[cid] += 1
                
    top_cands = [cid for cid, _ in counts.most_common(50) if cid != eid]
    cands_count += len(top_cands)
    start_idx = len(feature_rows)
    for cid in top_cands:
        cname, caddr, ccntry = target_records[cid]
        feature_rows.append(extract_pair_features(name, addr, cntry, cname, caddr, ccntry, cand_id=cid))
    offsets.append((eid, top_cands, start_idx, len(feature_rows)))

probs = model.predict_proba(np.array(feature_rows, dtype=np.float32))[:, 1] if feature_rows else np.array([])
matches = 0
for eid, top_cands, start_idx, end_idx in offsets:
    if not top_cands:
        continue
    p_sub = probs[start_idx:end_idx]
    p_max = float(np.max(p_sub))
    if p_max >= 0.985:
        acc = [cid for cid, p in zip(top_cands, p_sub) if p >= 0.985 and p >= p_max - 0.005]
        matches += len(acc)

duration = time.time() - t1
print(f"10,000 queries completed in {duration:.2f}s! ({10000/duration:.0f} queries/sec). Total candidate pairs: {cands_count:,}. Matches: {matches:,}")
