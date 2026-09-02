"""Detailed FP/FN error pattern classification for dissertation qualitative review."""
import pandas as pd
import os
import re
from collections import defaultdict, Counter

BASE = r"D:\Dissertation\file"
OUTPUT_DIR = os.path.join(BASE, "results_agentic")

med_meta = pd.read_csv(os.path.join(BASE, "med_metadata.csv"), dtype=str)
med_meta.columns = med_meta.columns.str.strip().str.lower()
med_valid_ids = set(med_meta["medcodeid"].dropna().str.strip().tolist())
med_lookup = dict(zip(med_meta["medcodeid"].str.strip(), med_meta["term"].fillna("")))

prod_meta = pd.read_csv(os.path.join(BASE, "prod_metadata.csv"), dtype=str)
prod_meta.columns = prod_meta.columns.str.strip().str.lower()
prod_valid_ids = set(prod_meta["prodcodeid"].dropna().str.strip().tolist())
prod_lookup = dict(zip(prod_meta["prodcodeid"].str.strip(), prod_meta["productname"].fillna("")))

def load_raw(filename, id_col="medcodeid"):
    df = pd.read_csv(os.path.join(BASE, filename), dtype=str)
    df.columns = df.columns.str.strip().str.lower()
    return set(df[id_col].dropna().str.strip().tolist())

def load_raw_lshtm(filename, sep="\t"):
    df = pd.read_csv(os.path.join(BASE, filename), dtype=str, sep=sep, engine="python")
    df.columns = df.columns.str.strip().str.lower()
    return set(df["medcodeid"].dropna().str.strip().tolist())

CONDITIONS = [
    {"name": "Atrial Fibrillation", "dict_type": "medical",
     "raw_loader": lambda: load_raw("AF-1_final_matched.csv"),
     "keywords": ["atrial fibrillation", "atrial flutter", "a-fib", "a fib", "af "]},
    {"name": "Heart Failure", "dict_type": "medical",
     "raw_loader": lambda: load_raw("heart failrue_final_matched.csv"),
     "keywords": ["heart failure", "cardiac failure", "ventricular failure", "cardiomyopathy",
                   "left ventricular", "right ventricular", "lvf", "rvf", "chf", "ccf"]},
    {"name": "Cerebrovascular Disease", "dict_type": "medical",
     "raw_loader": lambda: load_raw("cerebrovascular disease_final_matched.csv"),
     "keywords": ["stroke", "cerebrovascular", "cerebral infarction", "tia", "transient isch",
                   "subarachnoid", "intracerebral", "cerebral haemorrhage", "cerebral hemorrhage",
                   "carotid", "cerebral artery", "cerebral vein"]},
    {"name": "Hypertension", "dict_type": "medical",
     "raw_loader": lambda: load_raw_lshtm("hypertension.txt", "\t"),
     "keywords": ["hypertension", "hypertensive", "high blood pressure", "bp reading"]},
    {"name": "Myocardial Infarction", "dict_type": "medical",
     "raw_loader": lambda: load_raw_lshtm("myocardial infraction.txt", "\t"),
     "keywords": ["myocardial infarction", "mi ", "stemi", "nstemi", "heart attack",
                   "infarct", "coronary thrombosis"]},
    {"name": "Peripheral Arterial Disease", "dict_type": "medical",
     "raw_loader": lambda: load_raw_lshtm("peripheral.txt", "\t"),
     "keywords": ["peripheral", "claudication", "arterial disease", "arterial occlus",
                   "limb isch", "gangrene", "amputation", "femoral", "popliteal", "iliac"]},
    {"name": "COPD", "dict_type": "medical",
     "raw_loader": lambda: load_raw_lshtm("COPD.txt", "\t"),
     "keywords": ["copd", "chronic obstructive", "emphysema", "chronic bronchitis",
                   "obstructive airway", "obstructive pulmonary"]},
    {"name": "Asthma", "dict_type": "medical",
     "raw_loader": lambda: load_raw_lshtm("asthma.txt", ","),
     "keywords": ["asthma", "asthmatic", "bronchospasm", "wheez"]},
    {"name": "Insulin", "dict_type": "product",
     "raw_loader": lambda: load_raw("insulin_final_matched.csv", "prodcodeid"),
     "keywords": ["insulin"]},
    {"name": "Metformin", "dict_type": "product",
     "raw_loader": lambda: load_raw("metformin_final_matched.csv", "prodcodeid"),
     "keywords": ["metformin", "glucophage"]},
]

# ── FP classification rules ──
def classify_fp(term, condition_keywords):
    t = term.lower()

    # 1. Exclusion/ruled out
    if any(x in t for x in ["excluded", "ruled out", "not found", "no evidence",
                             "ecg: no ", "absent"]):
        return "Exclusion/ruled-out code"

    # 2. History/resolved/past
    if any(x in t for x in ["h/o:", "history of", "resolved", "past ", "personal history",
                             "[v]personal history"]):
        return "History/resolved code"

    # 3. Risk/screening/suspected
    if any(x in t for x in ["at risk", "increased risk", "screening", "suspected",
                             "risk assess", "risk factor", "risk score"]):
        return "Risk/screening/suspected"

    # 4. Administrative/monitoring
    if any(x in t for x in ["follow-up", "follow up", "annual review", "monitoring",
                             "clinic", "care pathway", "care plan", "review",
                             "referral", "discharge", "invitation", "exception",
                             "qof", "quality indicator", "admin", "dna -",
                             "did not attend", "declined", "seen in"]):
        return "Administrative/monitoring"

    # 5. Check if it's a valid diagnosis subtype
    if any(kw in t for kw in condition_keywords):
        return "Valid diagnosis subtype (reference gap)"

    # 6. Semantically related but different condition
    return "Semantic drift (related but different)"


# ── FN classification rules ──
def classify_fn(term, code_id, valid_ids):
    t = term.lower()

    # 1. Not in dictionary
    if code_id not in valid_ids:
        return "Not in dictionary"

    # 2. Administrative/monitoring
    if any(x in t for x in ["follow-up", "follow up", "annual review", "monitoring",
                             "clinic", "care pathway", "care plan", "review",
                             "referral", "discharge", "invitation", "exception",
                             "qof", "quality indicator", "admin", "dna -",
                             "did not attend", "declined", "seen in", "screen",
                             "letter", "leaflet", "advice", "education"]):
        return "Administrative/monitoring code"

    # 3. Procedural/surgical
    if any(x in t for x in ["percutaneous", "transluminal", "insertion", "stent",
                             "repair", "bypass", "graft", "surgical", "operation",
                             "angioplasty", "endarterectomy", "amputation",
                             "replacement", "anastomosis", "embolectomy",
                             "catheter", "thrombolysis", "resuscitation",
                             "therapeutic", "endovas"]):
        return "Procedural/surgical code"

    # 4. Cause of death
    if any(x in t for x in ["cause of death", "death cert"]):
        return "Cause-of-death code"

    # 5. History/resolved
    if any(x in t for x in ["h/o:", "history of", "resolved", "personal history",
                             "[v]personal history"]):
        return "History/resolved code"

    # 6. Sequelae
    if any(x in t for x in ["sequela", "late effect", "old infarct", "old "]):
        return "Sequelae/historical code"

    # 7. Complication code
    if any(x in t for x in ["complication", "as current complication"]):
        return "Complication code"

    # 8. Genuine miss
    return "Missed valid diagnosis"


MODELS = ["GPT-5.4_Mini", "Claude_Sonnet_5", "Gemini_3.6_Flash"]

# ── Run analysis ──
all_fp_summary = []
all_fn_summary = []
all_fp_examples = []

for c in CONDITIONS:
    name = c["name"]
    dict_type = c["dict_type"]
    lookup = med_lookup if dict_type == "medical" else prod_lookup
    valid_ids = med_valid_ids if dict_type == "medical" else prod_valid_ids
    raw_ref = c["raw_loader"]()
    ref_in_dict = raw_ref & valid_ids
    ref_not_in_dict = raw_ref - valid_ids

    print(f"\n{'=' * 90}")
    print(f"  {name}")
    print(f"  Ref: {len(raw_ref)} source → {len(ref_in_dict)} in dict, {len(ref_not_in_dict)} not in dict")
    print(f"{'=' * 90}")

    for model in MODELS:
        tag = f"{name}_{model}".replace(" ", "_")
        codelist_file = os.path.join(OUTPUT_DIR, f"{tag}_codelist.csv")
        if not os.path.exists(codelist_file):
            continue

        ai_df = pd.read_csv(codelist_file, dtype=str)
        ai_codes = set(ai_df["code_id"].dropna().str.strip().tolist()) & valid_ids

        fp_codes = ai_codes - ref_in_dict
        fn_codes_dict = ref_in_dict - ai_codes
        fn_codes_nodict = ref_not_in_dict

        # Classify FPs
        fp_categories = defaultdict(list)
        for cid in sorted(fp_codes):
            term = lookup.get(cid, "?")
            cat = classify_fp(term, c["keywords"])
            fp_categories[cat].append((cid, term))

        # Classify FNs (include not-in-dict)
        fn_categories = defaultdict(list)
        for cid in sorted(fn_codes_nodict):
            fn_categories["Not in dictionary"].append((cid, "N/A"))
        for cid in sorted(fn_codes_dict):
            term = lookup.get(cid, "?")
            cat = classify_fn(term, cid, valid_ids)
            fn_categories[cat].append((cid, term))

        print(f"\n  [{model}] AI={len(ai_codes)} | TP={len(ai_codes & ref_in_dict)} "
              f"FP={len(fp_codes)} FN={len(fn_codes_dict) + len(fn_codes_nodict)}")

        print(f"  FP breakdown:")
        for cat in ["Valid diagnosis subtype (reference gap)", "Semantic drift (related but different)",
                     "History/resolved code", "Risk/screening/suspected",
                     "Administrative/monitoring", "Exclusion/ruled-out code"]:
            items = fp_categories.get(cat, [])
            if items:
                print(f"    {cat}: {len(items)}")
                for cid, term in items[:3]:
                    print(f"      e.g. {term}")
                all_fp_examples.append({
                    "Condition": name, "Model": model, "Category": cat,
                    "Count": len(items),
                    "Examples": "; ".join([t for _, t in items[:3]])
                })
            all_fp_summary.append({
                "Condition": name, "Model": model,
                "FP_Category": cat, "Count": len(items)
            })

        print(f"  FN breakdown:")
        for cat in ["Not in dictionary", "Administrative/monitoring code",
                     "Procedural/surgical code", "Cause-of-death code",
                     "History/resolved code", "Sequelae/historical code",
                     "Complication code", "Missed valid diagnosis"]:
            items = fn_categories.get(cat, [])
            if items:
                print(f"    {cat}: {len(items)}")
                for cid, term in items[:3]:
                    print(f"      e.g. {term}")
            all_fn_summary.append({
                "Condition": name, "Model": model,
                "FN_Category": cat, "Count": len(items)
            })

# ── Save summary CSVs ──
pd.DataFrame(all_fp_summary).to_csv(
    os.path.join(OUTPUT_DIR, "fp_error_analysis.csv"), index=False)
pd.DataFrame(all_fn_summary).to_csv(
    os.path.join(OUTPUT_DIR, "fn_error_analysis.csv"), index=False)
pd.DataFrame(all_fp_examples).to_csv(
    os.path.join(OUTPUT_DIR, "fp_examples.csv"), index=False)

# ── Aggregate across all conditions per model ──
print("\n" + "=" * 90)
print("AGGREGATE FP PATTERNS (all conditions)")
print("=" * 90)
fp_df = pd.DataFrame(all_fp_summary)
for model in MODELS:
    mdf = fp_df[fp_df["Model"] == model]
    total_fp = mdf["Count"].sum()
    print(f"\n  {model} (total FP={total_fp}):")
    for cat in ["Valid diagnosis subtype (reference gap)", "Semantic drift (related but different)",
                 "History/resolved code", "Risk/screening/suspected",
                 "Administrative/monitoring", "Exclusion/ruled-out code"]:
        n = mdf[mdf["FP_Category"] == cat]["Count"].sum()
        pct = n / total_fp * 100 if total_fp else 0
        print(f"    {cat}: {n} ({pct:.1f}%)")

print("\n" + "=" * 90)
print("AGGREGATE FN PATTERNS (all conditions)")
print("=" * 90)
fn_df = pd.DataFrame(all_fn_summary)
for model in MODELS:
    mdf = fn_df[fn_df["Model"] == model]
    total_fn = mdf["Count"].sum()
    print(f"\n  {model} (total FN={total_fn}):")
    for cat in ["Not in dictionary", "Administrative/monitoring code",
                 "Procedural/surgical code", "Cause-of-death code",
                 "History/resolved code", "Sequelae/historical code",
                 "Complication code", "Missed valid diagnosis"]:
        n = mdf[mdf["FN_Category"] == cat]["Count"].sum()
        pct = n / total_fn * 100 if total_fn else 0
        print(f"    {cat}: {n} ({pct:.1f}%)")

print("\nDONE — saved fp_error_analysis.csv, fn_error_analysis.csv, fp_examples.csv")
