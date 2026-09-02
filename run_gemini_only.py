"""Re-run only Gemini 3.6 Flash using native google-genai SDK.
Merges results into the existing evaluation CSV.
"""
import pandas as pd
import numpy as np
import os
import json
import time
from openai import OpenAI
from google import genai
from google.genai import types as gtypes

BASE = r"D:\Dissertation\file"
OUTPUT_DIR = os.path.join(BASE, "results_agentic")
EMBED_MODEL = "text-embedding-3-large"
EMBED_DIM = 1536
K_DEFAULT = 50
K_MAX = 200
MAX_ITERATIONS = 20

openai_client = OpenAI()
gemini_client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

MODEL_NAME = "Gemini_3.6_Flash"
MODEL_ID = "gemini-3.6-flash"

print("Loading pre-computed embeddings...")
med_emb = np.load(os.path.join(BASE, "med_embeddings.npy"))
med_meta = pd.read_csv(os.path.join(BASE, "med_metadata.csv"), dtype=str)
prod_emb = np.load(os.path.join(BASE, "prod_embeddings.npy"))
prod_meta = pd.read_csv(os.path.join(BASE, "prod_metadata.csv"), dtype=str)

med_norms = np.linalg.norm(med_emb, axis=1, keepdims=True)
med_norms[med_norms == 0] = 1
med_emb_norm = med_emb / med_norms

prod_norms = np.linalg.norm(prod_emb, axis=1, keepdims=True)
prod_norms[prod_norms == 0] = 1
prod_emb_norm = prod_emb / prod_norms

med_valid_ids = set(med_meta["MedCodeId"].tolist())
prod_valid_ids = set(prod_meta["ProdCodeId"].tolist())
print(f"  Medical: {med_emb.shape[0]} codes | Product: {prod_emb.shape[0]} codes")

def load_text_matched(filename, id_col="medcodeid"):
    df = pd.read_csv(os.path.join(BASE, filename), dtype=str)
    df.columns = df.columns.str.strip().str.lower()
    return set(df[id_col].dropna().str.strip().tolist())

def load_lshtm(filename, sep="\t"):
    df = pd.read_csv(os.path.join(BASE, filename), dtype=str, sep=sep, engine="python")
    df.columns = df.columns.str.strip().str.lower()
    codes = set(df["medcodeid"].dropna().str.strip().tolist())
    return codes & med_valid_ids

def load_hdr_uk(filename):
    df = pd.read_csv(os.path.join(BASE, filename), dtype=str)
    df.columns = df.columns.str.strip().str.lower()
    codes = set(df["code"].dropna().str.strip().tolist())
    return codes & med_valid_ids

CONDITIONS = [
    {"name": "Atrial Fibrillation", "dict_type": "medical",
     "ref_loader": lambda: load_text_matched("AF-1_final_matched.csv")},
    {"name": "Heart Failure", "dict_type": "medical",
     "ref_loader": lambda: load_text_matched("heart failrue_final_matched.csv")},
    {"name": "Cerebrovascular Disease", "dict_type": "medical",
     "ref_loader": lambda: load_text_matched("cerebrovascular disease_final_matched.csv")},
    {"name": "Hypertension", "dict_type": "medical",
     "ref_loader": lambda: load_lshtm("hypertension.txt", "\t")},
    {"name": "Myocardial Infarction", "dict_type": "medical",
     "ref_loader": lambda: load_lshtm("myocardial infraction.txt", "\t")},
    {"name": "Peripheral Arterial Disease", "dict_type": "medical",
     "ref_loader": lambda: load_lshtm("peripheral.txt", "\t")},
    {"name": "COPD", "dict_type": "medical",
     "ref_loader": lambda: load_lshtm("COPD.txt", "\t")},
    {"name": "Asthma", "dict_type": "medical",
     "ref_loader": lambda: load_lshtm("asthma.txt", ",")},
    {"name": "Insulin", "dict_type": "product",
     "ref_loader": lambda: load_text_matched("insulin_final_matched.csv", "prodcodeid")},
    {"name": "Metformin", "dict_type": "product",
     "ref_loader": lambda: load_text_matched("metformin_final_matched.csv", "prodcodeid")},
]

def search_dictionary(query, dict_type, k=K_DEFAULT):
    if dict_type == "medical":
        emb_norm, meta, id_col, term_col = med_emb_norm, med_meta, "MedCodeId", "Term"
    else:
        emb_norm, meta, id_col, term_col = prod_emb_norm, prod_meta, "ProdCodeId", "ProductName"
    resp = openai_client.embeddings.create(model=EMBED_MODEL, input=[query], dimensions=EMBED_DIM)
    q = np.array(resp.data[0].embedding, dtype=np.float32)
    q = q / np.linalg.norm(q)
    sims = emb_norm @ q
    top_idx = np.argsort(sims)[-min(k, K_MAX):][::-1]
    results = []
    for idx in top_idx:
        results.append({
            "code_id": meta.iloc[idx][id_col],
            "term": str(meta.iloc[idx][term_col]),
            "similarity": round(float(sims[idx]), 4)
        })
    return results

MEDICAL_SYSTEM = """You are a clinical coding expert building a codelist for {condition} using the CPRD Aurum medical dictionary.
Include a code if it represents a current diagnosis of {condition}.
Search for the condition, its subtypes, and synonyms. You MUST call submit_codelist with all included code IDs when done. Never output code IDs as text."""

PRODUCT_SYSTEM = """You are a clinical coding expert building a codelist for {condition} products using the CPRD Aurum product dictionary.
Include a code if it is a {condition} preparation or contains {condition} as an active ingredient.
Search for the product and its variants. You MUST call submit_codelist with all included code IDs when done. Never output code IDs as text."""

def get_system_prompt(condition_name, dict_type):
    template = PRODUCT_SYSTEM if dict_type == "product" else MEDICAL_SYSTEM
    return template.format(condition=condition_name)

GEMINI_TOOLS = [gtypes.Tool(function_declarations=[
    gtypes.FunctionDeclaration(
        name="search_dictionary",
        description="Search the CPRD Aurum dictionary for codes matching a query. Returns the most similar codes.",
        parameters=gtypes.Schema(
            type="OBJECT",
            properties={
                "query": gtypes.Schema(type="STRING", description="Search query term"),
                "k": gtypes.Schema(type="INTEGER", description="Number of results (default 50, max 200)")
            },
            required=["query"]
        )
    ),
    gtypes.FunctionDeclaration(
        name="submit_codelist",
        description="Submit the final codelist when done. Pass all code IDs to include.",
        parameters=gtypes.Schema(
            type="OBJECT",
            properties={
                "code_ids": gtypes.Schema(
                    type="ARRAY",
                    items=gtypes.Schema(type="STRING"),
                    description="All code IDs to include"
                )
            },
            required=["code_ids"]
        )
    )
])]

def run_agentic_gemini(model_id, system_prompt, dict_type):
    config = gtypes.GenerateContentConfig(
        tools=GEMINI_TOOLS,
        system_instruction=system_prompt,
        max_output_tokens=8000,
    )
    contents = [gtypes.Content(role="user", parts=[gtypes.Part(text="Build the codelist.")])]

    total_input = 0
    total_output = 0
    search_count = 0
    nudge_count = 0
    log_lines = []

    for iteration in range(MAX_ITERATIONS + 3):
        for retry in range(5):
            try:
                resp = gemini_client.models.generate_content(
                    model=model_id, contents=contents, config=config
                )
                break
            except Exception as e:
                if "429" in str(e) and retry < 4:
                    wait = 2 ** (retry + 1)
                    log_lines.append(f"[iter {iteration}] Rate limited, waiting {wait}s...")
                    time.sleep(wait)
                    continue
                log_lines.append(f"ERROR iter {iteration}: {e}")
                resp = None
                break
        if resp is None:
            break

        if resp.usage_metadata:
            total_input += resp.usage_metadata.prompt_token_count or 0
            total_output += resp.usage_metadata.candidates_token_count or 0

        assistant_content = resp.candidates[0].content
        contents.append(assistant_content)

        function_calls = [p for p in assistant_content.parts if p.function_call]

        if not function_calls:
            if search_count > 0 and nudge_count < 2:
                log_lines.append(f"[iter {iteration}] No tool calls — nudging to submit.")
                contents.append(gtypes.Content(role="user", parts=[
                    gtypes.Part(text="Now call submit_codelist with all the code IDs you want to include.")
                ]))
                nudge_count += 1
                continue
            log_lines.append(f"[iter {iteration}] No tool calls, stopping.")
            break

        fn_response_parts = []
        submitted = None

        for part in function_calls:
            fc = part.function_call
            if fc.name == "search_dictionary":
                query = fc.args.get("query", "")
                k = min(int(fc.args.get("k", K_DEFAULT)), K_MAX)
                results = search_dictionary(query, dict_type, k)
                search_count += 1
                log_lines.append(f"[iter {iteration}] search({query!r}, k={k}) => {len(results)} results")
                fn_response_parts.append(gtypes.Part(
                    function_response=gtypes.FunctionResponse(
                        name="search_dictionary",
                        response={"results": json.dumps(results)}
                    )
                ))
            elif fc.name == "submit_codelist":
                code_ids = fc.args.get("code_ids", [])
                log_lines.append(f"[iter {iteration}] submit_codelist({len(code_ids)} codes)")
                submitted = list(code_ids)

        if submitted is not None:
            return {
                "code_ids": set(submitted),
                "search_count": search_count,
                "iterations": iteration + 1,
                "input_tokens": total_input,
                "output_tokens": total_output,
                "log": "\n".join(log_lines)
            }

        contents.append(gtypes.Content(role="user", parts=fn_response_parts))
        time.sleep(0.5)

    log_lines.append("NO SUBMIT — loop ended")
    return {
        "code_ids": set(),
        "search_count": search_count,
        "iterations": iteration + 1 if 'iteration' in dir() else 0,
        "input_tokens": total_input,
        "output_tokens": total_output,
        "log": "\n".join(log_lines),
        "error": "no_submit"
    }

def evaluate(ai_codes, ref_codes):
    tp = len(ai_codes & ref_codes)
    fp = len(ai_codes - ref_codes)
    fn = len(ref_codes - ai_codes)
    prec = tp / (tp + fp) if (tp + fp) else 0
    rec = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
    jacc = tp / (tp + fp + fn) if (tp + fp + fn) else 0
    return {"TP": tp, "FP": fp, "FN": fn,
            "Precision": round(prec, 4), "Recall": round(rec, 4),
            "F1": round(f1, 4), "Jaccard": round(jacc, 4),
            "AI_size": len(ai_codes), "Ref_size": len(ref_codes)}

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN — Gemini only
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*80}")
print(f"GEMINI-ONLY RE-RUN (native SDK)")
print(f"{'='*80}")

gemini_results = []

existing_csv = os.path.join(OUTPUT_DIR, "evaluation_results_all.csv")
skip_conditions = set()
if os.path.exists(existing_csv):
    edf = pd.read_csv(existing_csv)
    gemini_ok = edf[(edf["Model"] == MODEL_NAME) & (edf["F1"] > 0)]
    skip_conditions = set(gemini_ok["Condition"].tolist())
    if skip_conditions:
        print(f"  Skipping already-successful: {skip_conditions}")

for cond in CONDITIONS:
    if cond["name"] in skip_conditions:
        print(f"\n  {cond['name']} | SKIPPED (already F1>0)")
        continue

    ref_codes = cond["ref_loader"]()
    print(f"\n  {cond['name']} | ref={len(ref_codes)} | dict={cond['dict_type']}")

    system_prompt = get_system_prompt(cond["name"], cond["dict_type"])
    print(f"  [{MODEL_NAME}] agentic loop...", end=" ", flush=True)
    t0 = time.time()

    try:
        result = run_agentic_gemini(MODEL_ID, system_prompt, cond["dict_type"])
        dt = time.time() - t0

        valid_ids = med_valid_ids if cond["dict_type"] == "medical" else prod_valid_ids
        ai_codes = result["code_ids"] & valid_ids
        metrics = evaluate(ai_codes, ref_codes)

        print(f"{dt:.1f}s | {result['search_count']} searches | "
              f"AI={metrics['AI_size']} | "
              f"P={metrics['Precision']:.3f} R={metrics['Recall']:.3f} "
              f"F1={metrics['F1']:.3f} | "
              f"tok={result['input_tokens']}in/{result['output_tokens']}out")

        row = {
            "Condition": cond["name"], "Model": MODEL_NAME,
            "N_searches": result["search_count"],
            "N_iterations": result["iterations"],
            "Input_tokens": result["input_tokens"],
            "Output_tokens": result["output_tokens"],
            **metrics
        }
        if "error" in result:
            row["Error"] = result["error"]
        gemini_results.append(row)

        tag = f"{cond['name']}_{MODEL_NAME}".replace(" ", "_")
        pd.DataFrame({"code_id": sorted(ai_codes)}).to_csv(
            os.path.join(OUTPUT_DIR, f"{tag}_codelist.csv"), index=False)
        with open(os.path.join(OUTPUT_DIR, f"{tag}_log.txt"), "w", encoding="utf-8") as f:
            f.write(result["log"])

    except Exception as e:
        dt = time.time() - t0
        print(f"ERROR ({dt:.1f}s): {str(e)[:200]}")
        gemini_results.append({
            "Condition": cond["name"], "Model": MODEL_NAME,
            "Error": str(e)[:200]
        })

    time.sleep(1)

# Merge with existing results — keep successful rows, replace failed ones
if os.path.exists(existing_csv):
    existing = pd.read_csv(existing_csv)
    new_conditions = {r["Condition"] for r in gemini_results}
    keep = existing[~((existing["Model"] == MODEL_NAME) & (existing["Condition"].isin(new_conditions)))]
    merged = pd.concat([keep, pd.DataFrame(gemini_results)], ignore_index=True)
else:
    merged = pd.DataFrame(gemini_results)

merged.to_csv(existing_csv, index=False)
print(f"\nMerged results saved to {existing_csv}")
print(f"Total rows: {len(merged)}")

gemini_df = pd.DataFrame(gemini_results)
if "F1" in gemini_df.columns:
    print(f"\n  Gemini 3.6 Flash mean F1: {gemini_df['F1'].mean():.3f}")
    print(f"  Gemini 3.6 Flash mean searches: {gemini_df['N_searches'].mean():.1f}")

print("\nDONE — Gemini re-run complete")
