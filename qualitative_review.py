"""Check ref dictionary coverage + qualitative review of AI FP/FN codes."""
import pandas as pd
import os

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
     "raw_loader": lambda: load_raw("AF-1_final_matched.csv"), "valid_ids": med_valid_ids},
    {"name": "Heart Failure", "dict_type": "medical",
     "raw_loader": lambda: load_raw("heart failrue_final_matched.csv"), "valid_ids": med_valid_ids},
    {"name": "Cerebrovascular Disease", "dict_type": "medical",
     "raw_loader": lambda: load_raw("cerebrovascular disease_final_matched.csv"), "valid_ids": med_valid_ids},
    {"name": "Hypertension", "dict_type": "medical",
     "raw_loader": lambda: load_raw_lshtm("hypertension.txt", "\t"), "valid_ids": med_valid_ids},
    {"name": "Myocardial Infarction", "dict_type": "medical",
     "raw_loader": lambda: load_raw_lshtm("myocardial infraction.txt", "\t"), "valid_ids": med_valid_ids},
    {"name": "Peripheral Arterial Disease", "dict_type": "medical",
     "raw_loader": lambda: load_raw_lshtm("peripheral.txt", "\t"), "valid_ids": med_valid_ids},
    {"name": "COPD", "dict_type": "medical",
     "raw_loader": lambda: load_raw_lshtm("COPD.txt", "\t"), "valid_ids": med_valid_ids},
    {"name": "Asthma", "dict_type": "medical",
     "raw_loader": lambda: load_raw_lshtm("asthma.txt", ","), "valid_ids": med_valid_ids},
    {"name": "Insulin", "dict_type": "product",
     "raw_loader": lambda: load_raw("insulin_final_matched.csv", "prodcodeid"), "valid_ids": prod_valid_ids},
    {"name": "Metformin", "dict_type": "product",
     "raw_loader": lambda: load_raw("metformin_final_matched.csv", "prodcodeid"), "valid_ids": prod_valid_ids},
]

print("=" * 90)
print("PART 1: REFERENCE CODELIST — DICTIONARY COVERAGE")
print("=" * 90)
print(f"{'Condition':<30} {'Source':>12} {'In Dict':>10} {'Not in Dict':>12} {'Coverage%':>10}")
print("-" * 90)

ref_data = {}
for c in CONDITIONS:
    raw = c["raw_loader"]()
    in_dict = raw & c["valid_ids"]
    not_in = raw - c["valid_ids"]
    pct = len(in_dict) / len(raw) * 100 if raw else 0
    print(f"{c['name']:<30} {len(raw):>12} {len(in_dict):>10} {len(not_in):>12} {pct:>9.1f}%")
    ref_data[c["name"]] = {"raw": raw, "in_dict": in_dict, "not_in": not_in}

# ── Part 2: Qualitative review ──
print("\n" + "=" * 90)
print("PART 2: QUALITATIVE REVIEW — SAMPLE FP AND FN CODES")
print("=" * 90)

MODELS = ["GPT-5.4_Mini", "Claude_Sonnet_5", "Gemini_3.6_Flash"]

for c in CONDITIONS:
    name = c["name"]
    dict_type = c["dict_type"]
    lookup = med_lookup if dict_type == "medical" else prod_lookup
    ref_codes = ref_data[name]["in_dict"]

    print(f"\n{'=' * 90}")
    print(f"  {name} | Ref in dict: {len(ref_codes)} | Ref not in dict: {len(ref_data[name]['not_in'])}")
    print(f"{'=' * 90}")

    for model in MODELS:
        tag = f"{name}_{model}".replace(" ", "_")
        codelist_file = os.path.join(OUTPUT_DIR, f"{tag}_codelist.csv")
        if not os.path.exists(codelist_file):
            print(f"\n  [{model}] codelist not found")
            continue
        ai_df = pd.read_csv(codelist_file, dtype=str)
        ai_codes = set(ai_df["code_id"].dropna().str.strip().tolist())

        valid = med_valid_ids if dict_type == "medical" else prod_valid_ids
        ai_codes = ai_codes & valid

        tp_codes = ai_codes & ref_codes
        fp_codes = ai_codes - ref_codes
        fn_codes = ref_codes - ai_codes

        print(f"\n  [{model}] AI={len(ai_codes)} | TP={len(tp_codes)} FP={len(fp_codes)} FN={len(fn_codes)}")

        if fp_codes:
            sample = sorted(fp_codes)[:20]
            print(f"  FP sample (AI included, NOT in reference — potentially valid?):")
            for i, cid in enumerate(sample):
                term = lookup.get(cid, "?")
                print(f"    {i+1}. {cid} | {term}")
            if len(fp_codes) > 20:
                print(f"    ... +{len(fp_codes) - 20} more")

        if fn_codes:
            sample = sorted(fn_codes)[:10]
            print(f"  FN sample (in reference, AI missed):")
            for i, cid in enumerate(sample):
                term = lookup.get(cid, "?")
                print(f"    {i+1}. {cid} | {term}")
            if len(fn_codes) > 10:
                print(f"    ... +{len(fn_codes) - 10} more")

print("\nDONE")
