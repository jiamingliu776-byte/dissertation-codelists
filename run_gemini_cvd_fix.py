"""Re-run only CVD for Gemini 3.6 Flash with fallback submission.
When the main loop ends without submit, a second pass presents all
collected search results and asks the model to select and submit.
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

med_norms = np.linalg.norm(med_emb, axis=1, keepdims=True)
med_norms[med_norms == 0] = 1
med_emb_norm = med_emb / med_norms

med_valid_ids = set(med_meta["MedCodeId"].tolist())
print(f"  Medical: {med_emb.shape[0]} codes")

def load_text_matched(filename, id_col="medcodeid"):
    df = pd.read_csv(os.path.join(BASE, filename), dtype=str)
    df.columns = df.columns.str.strip().str.lower()
    return set(df[id_col].dropna().str.strip().tolist())

def search_dictionary(query, k=K_DEFAULT):
    resp = openai_client.embeddings.create(model=EMBED_MODEL, input=[query], dimensions=EMBED_DIM)
    q = np.array(resp.data[0].embedding, dtype=np.float32)
    q = q / np.linalg.norm(q)
    sims = med_emb_norm @ q
    top_idx = np.argsort(sims)[-min(k, K_MAX):][::-1]
    results = []
    for idx in top_idx:
        results.append({
            "code_id": med_meta.iloc[idx]["MedCodeId"],
            "term": str(med_meta.iloc[idx]["Term"]),
            "similarity": round(float(sims[idx]), 4)
        })
    return results

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

SUBMIT_ONLY_TOOLS = [gtypes.Tool(function_declarations=[
    gtypes.FunctionDeclaration(
        name="submit_codelist",
        description="Submit the final codelist. Pass all code IDs to include.",
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

SYSTEM_PROMPT = """You are a clinical coding expert building a codelist for Cerebrovascular Disease using the CPRD Aurum medical dictionary.
Include a code if it represents a current diagnosis of Cerebrovascular Disease.
Search for the condition, its subtypes, and synonyms. You MUST call submit_codelist with all included code IDs when done. Never output code IDs as text."""

def run_with_fallback():
    config = gtypes.GenerateContentConfig(
        tools=GEMINI_TOOLS,
        system_instruction=SYSTEM_PROMPT,
        max_output_tokens=8000,
    )
    contents = [gtypes.Content(role="user", parts=[gtypes.Part(text="Build the codelist.")])]

    total_input = 0
    total_output = 0
    search_count = 0
    nudge_count = 0
    log_lines = []
    all_search_results = {}  # code_id -> {term, max_similarity}

    for iteration in range(MAX_ITERATIONS + 3):
        for retry in range(5):
            try:
                resp = gemini_client.models.generate_content(
                    model=MODEL_ID, contents=contents, config=config
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
                results = search_dictionary(query, k)
                search_count += 1
                log_lines.append(f"[iter {iteration}] search({query!r}, k={k}) => {len(results)} results")
                for r in results:
                    cid = r["code_id"]
                    if cid not in all_search_results or r["similarity"] > all_search_results[cid]["similarity"]:
                        all_search_results[cid] = {"term": r["term"], "similarity": r["similarity"]}
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
                "input_tokens": total_input,
                "output_tokens": total_output,
                "log": "\n".join(log_lines),
                "fallback": False
            }

        contents.append(gtypes.Content(role="user", parts=fn_response_parts))
        time.sleep(0.5)

    # ── FALLBACK PASS ──
    log_lines.append(f"NO SUBMIT after {search_count} searches — starting fallback pass")
    log_lines.append(f"  Collected {len(all_search_results)} unique codes from all searches")

    top_candidates = sorted(all_search_results.items(), key=lambda x: x[1]["similarity"], reverse=True)[:500]
    candidate_text = "\n".join(
        f"  {cid} | {info['term']} | sim={info['similarity']}"
        for cid, info in top_candidates
    )

    fallback_prompt = f"""Below are {len(top_candidates)} candidate codes from the CPRD Aurum medical dictionary, found by searching for cerebrovascular disease and related terms.

Review them and call submit_codelist with the code IDs that represent a current diagnosis of Cerebrovascular Disease (including stroke, TIA, cerebral infarction, subarachnoid hemorrhage, intracerebral hemorrhage, and other cerebrovascular conditions).

Candidates:
{candidate_text}

You MUST call submit_codelist now with the relevant code IDs."""

    fallback_config = gtypes.GenerateContentConfig(
        tools=SUBMIT_ONLY_TOOLS,
        system_instruction="You are a clinical coding expert. Select codes for Cerebrovascular Disease and call submit_codelist.",
        max_output_tokens=8000,
    )
    fallback_contents = [gtypes.Content(role="user", parts=[gtypes.Part(text=fallback_prompt)])]

    for attempt in range(3):
        log_lines.append(f"  Fallback attempt {attempt + 1}...")
        for retry in range(5):
            try:
                resp = gemini_client.models.generate_content(
                    model=MODEL_ID, contents=fallback_contents, config=fallback_config
                )
                break
            except Exception as e:
                if "429" in str(e) and retry < 4:
                    wait = 2 ** (retry + 1)
                    time.sleep(wait)
                    continue
                log_lines.append(f"  Fallback ERROR: {e}")
                resp = None
                break
        if resp is None:
            continue

        if resp.usage_metadata:
            total_input += resp.usage_metadata.prompt_token_count or 0
            total_output += resp.usage_metadata.candidates_token_count or 0

        assistant_content = resp.candidates[0].content
        fallback_contents.append(assistant_content)

        fcs = [p for p in assistant_content.parts if p.function_call]
        for part in fcs:
            fc = part.function_call
            if fc.name == "submit_codelist":
                code_ids = fc.args.get("code_ids", [])
                log_lines.append(f"  Fallback submit_codelist({len(code_ids)} codes) — SUCCESS")
                return {
                    "code_ids": set(code_ids),
                    "search_count": search_count,
                    "input_tokens": total_input,
                    "output_tokens": total_output,
                    "log": "\n".join(log_lines),
                    "fallback": True
                }

        fallback_contents.append(gtypes.Content(role="user", parts=[
            gtypes.Part(text="You MUST call submit_codelist now with the cerebrovascular disease code IDs.")
        ]))
        time.sleep(1)

    log_lines.append("  Fallback also failed to submit")
    return {
        "code_ids": set(),
        "search_count": search_count,
        "input_tokens": total_input,
        "output_tokens": total_output,
        "log": "\n".join(log_lines),
        "fallback": True,
        "error": "no_submit"
    }

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
ref_codes = load_text_matched("cerebrovascular disease_final_matched.csv")
print(f"\nCerebrovascular Disease | ref={len(ref_codes)} codes")
print(f"[{MODEL_NAME}] agentic loop with fallback...", flush=True)

t0 = time.time()
result = run_with_fallback()
dt = time.time() - t0

ai_codes = result["code_ids"] & med_valid_ids
tp = len(ai_codes & ref_codes)
fp = len(ai_codes - ref_codes)
fn = len(ref_codes - ai_codes)
prec = tp / (tp + fp) if (tp + fp) else 0
rec = tp / (tp + fn) if (tp + fn) else 0
f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
jacc = tp / (tp + fp + fn) if (tp + fp + fn) else 0

print(f"  {dt:.1f}s | {result['search_count']} searches | fallback={'Yes' if result.get('fallback') else 'No'}")
print(f"  AI={len(ai_codes)} | P={prec:.3f} R={rec:.3f} F1={f1:.3f} Jaccard={jacc:.3f}")
print(f"  tok={result['input_tokens']}in/{result['output_tokens']}out")

# Save codelist and log
pd.DataFrame({"code_id": sorted(ai_codes)}).to_csv(
    os.path.join(OUTPUT_DIR, "Cerebrovascular_Disease_Gemini_3.6_Flash_codelist.csv"), index=False)
with open(os.path.join(OUTPUT_DIR, "Cerebrovascular_Disease_Gemini_3.6_Flash_log.txt"), "w", encoding="utf-8") as f:
    f.write(result["log"])

# Update evaluation CSV
row = {
    "Condition": "Cerebrovascular Disease", "Model": MODEL_NAME,
    "N_searches": result["search_count"],
    "N_iterations": 0,
    "Input_tokens": result["input_tokens"],
    "Output_tokens": result["output_tokens"],
    "TP": tp, "FP": fp, "FN": fn,
    "Precision": round(prec, 4), "Recall": round(rec, 4),
    "F1": round(f1, 4), "Jaccard": round(jacc, 4),
    "AI_size": len(ai_codes), "Ref_size": len(ref_codes),
}

existing_csv = os.path.join(OUTPUT_DIR, "evaluation_results_all.csv")
existing = pd.read_csv(existing_csv)
keep = existing[~((existing["Model"] == MODEL_NAME) & (existing["Condition"] == "Cerebrovascular Disease"))]
merged = pd.concat([keep, pd.DataFrame([row])], ignore_index=True)
merged.to_csv(existing_csv, index=False)
print(f"\nUpdated {existing_csv}")
print("DONE")
