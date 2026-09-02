"""Agentic RAG Pipeline for clinical codelist generation.
The LLM autonomously decides search queries and when to stop.
"""
import pandas as pd
import numpy as np
import os
import json
import re
import time
from openai import OpenAI
from anthropic import Anthropic
from google import genai
from google.genai import types as gtypes

BASE = r"D:\Dissertation\file"
OUTPUT_DIR = os.path.join(BASE, "results_agentic")
os.makedirs(OUTPUT_DIR, exist_ok=True)

EMBED_MODEL = "text-embedding-3-large"
EMBED_DIM = 1536
K_DEFAULT = 50
K_MAX = 200
MAX_ITERATIONS = 20

openai_client = OpenAI()
anthropic_client = Anthropic(
    api_key=os.environ.get("ANTHROPIC_API_KEY"),
    default_headers={"anthropic-workspace-id": os.environ.get("ANTHROPIC_WORKSPACE_ID", "")}
)
gemini_client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

MODELS = {
    "GPT-5.4_Mini": {"client": "openai", "model_id": "gpt-5.4-mini"},
    "Claude_Sonnet_5": {"client": "anthropic", "model_id": "claude-sonnet-5"},
    "Gemini_3.6_Flash": {"client": "gemini", "model_id": "gemini-3.6-flash"},
}

# ═══════════════════════════════════════════════════════════════════════════════
# LOAD EMBEDDINGS
# ═══════════════════════════════════════════════════════════════════════════════
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

# ═══════════════════════════════════════════════════════════════════════════════
# REFERENCE CODELIST LOADERS
# ═══════════════════════════════════════════════════════════════════════════════
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

# ═══════════════════════════════════════════════════════════════════════════════
# CONDITIONS
# ═══════════════════════════════════════════════════════════════════════════════
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

AF_EXPERT_LISTS = [
    {"name": "AF-1", "loader": lambda: load_text_matched("AF-1_final_matched.csv")},
    {"name": "AF-2", "loader": lambda: load_hdr_uk("AF-2.csv")},
    {"name": "AF-3", "loader": lambda: load_lshtm("AF-3.txt", ",")},
    {"name": "AF-4", "loader": lambda: load_lshtm("AF-4.txt", "\t")},
    {"name": "AF-5", "loader": lambda: load_lshtm("AF-5.txt", "\t")},
]

# ═══════════════════════════════════════════════════════════════════════════════
# SEARCH FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════
def search_dictionary(query, dict_type, k=K_DEFAULT):
    if dict_type == "medical":
        emb_norm, meta, id_col, term_col = med_emb_norm, med_meta, "MedCodeId", "Term"
    else:
        emb_norm, meta, id_col, term_col = prod_emb_norm, prod_meta, "ProdCodeId", "ProductName"

    resp = openai_client.embeddings.create(
        model=EMBED_MODEL, input=[query], dimensions=EMBED_DIM
    )
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

# ═══════════════════════════════════════════════════════════════════════════════
# TOOL DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════════════
OPENAI_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_dictionary",
            "description": "Search the CPRD Aurum dictionary for codes matching a query. Returns the most similar codes.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query term"},
                    "k": {"type": "integer", "description": "Number of results (default 50, max 200)"}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "submit_codelist",
            "description": "Submit the final codelist when done. Pass all code IDs to include.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "All code IDs to include"
                    }
                },
                "required": ["code_ids"]
            }
        }
    }
]

ANTHROPIC_TOOLS = [
    {
        "name": "search_dictionary",
        "description": "Search the CPRD Aurum dictionary for codes matching a query. Returns the most similar codes.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query term"},
                "k": {"type": "integer", "description": "Number of results (default 50, max 200)"}
            },
            "required": ["query"]
        }
    },
    {
        "name": "submit_codelist",
        "description": "Submit the final codelist when done. Pass all code IDs to include.",
        "input_schema": {
            "type": "object",
            "properties": {
                "code_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "All code IDs to include"
                }
            },
            "required": ["code_ids"]
        }
    }
]

# ═══════════════════════════════════════════════════════════════════════════════
# PROMPTS (3 lines each)
# ═══════════════════════════════════════════════════════════════════════════════
MEDICAL_SYSTEM = """You are a clinical coding expert building a codelist for {condition} using the CPRD Aurum medical dictionary.
Include a code if it represents a current diagnosis of {condition}.
Search for the condition, its subtypes, and synonyms. You MUST call submit_codelist with all included code IDs when done. Never output code IDs as text."""

PRODUCT_SYSTEM = """You are a clinical coding expert building a codelist for {condition} products using the CPRD Aurum product dictionary.
Include a code if it is a {condition} preparation or contains {condition} as an active ingredient.
Search for the product and its variants. You MUST call submit_codelist with all included code IDs when done. Never output code IDs as text."""

def get_system_prompt(condition_name, dict_type):
    template = PRODUCT_SYSTEM if dict_type == "product" else MEDICAL_SYSTEM
    return template.format(condition=condition_name)

# ═══════════════════════════════════════════════════════════════════════════════
# AGENTIC LOOP — OPENAI COMPATIBLE (GPT-5 Mini, DeepSeek V4 Flash)
# ═══════════════════════════════════════════════════════════════════════════════
def run_agentic_openai(client, model_id, system_prompt, dict_type):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "Build the codelist."}
    ]

    total_input = 0
    total_output = 0
    search_count = 0
    nudge_count = 0
    log_lines = []
    use_max_completion_tokens = ("gpt-5" in model_id or "gpt-4o" in model_id)

    for iteration in range(MAX_ITERATIONS):
        try:
            api_kwargs = {
                "model": model_id,
                "messages": messages,
                "tools": OPENAI_TOOLS,
            }
            if use_max_completion_tokens:
                api_kwargs["max_completion_tokens"] = 8000
            else:
                api_kwargs["max_tokens"] = 8000
            resp = client.chat.completions.create(**api_kwargs)
        except Exception as e:
            log_lines.append(f"ERROR iter {iteration}: {e}")
            break

        if resp.usage:
            total_input += resp.usage.prompt_tokens
            total_output += resp.usage.completion_tokens

        msg = resp.choices[0].message

        msg_dict = {"role": "assistant", "content": msg.content or ""}
        if msg.tool_calls:
            msg_dict["tool_calls"] = [
                {"id": tc.id, "type": "function",
                 "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in msg.tool_calls
            ]
        messages.append(msg_dict)

        if not msg.tool_calls:
            if search_count > 0 and nudge_count < 2:
                log_lines.append(f"[iter {iteration}] No tool calls — nudging to submit.")
                messages.append({"role": "user",
                                 "content": "Now call submit_codelist with all the code IDs you want to include."})
                nudge_count += 1
                continue
            log_lines.append(f"[iter {iteration}] No tool calls, stopping.")
            break

        tool_responses = []
        submitted = None

        for tc in msg.tool_calls:
            fn = tc.function.name
            try:
                args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                log_lines.append(f"[iter {iteration}] Bad JSON args: {tc.function.arguments[:100]}")
                tool_responses.append({
                    "role": "tool", "tool_call_id": tc.id,
                    "content": '{"error": "invalid arguments"}'
                })
                continue

            if fn == "search_dictionary":
                query = args.get("query", "")
                k = min(args.get("k", K_DEFAULT), K_MAX)
                results = search_dictionary(query, dict_type, k)
                search_count += 1
                log_lines.append(f"[iter {iteration}] search({query!r}, k={k}) => {len(results)} results")
                tool_responses.append({
                    "role": "tool", "tool_call_id": tc.id,
                    "content": json.dumps(results)
                })

            elif fn == "submit_codelist":
                code_ids = args.get("code_ids", [])
                log_lines.append(f"[iter {iteration}] submit_codelist({len(code_ids)} codes)")
                submitted = code_ids

        if submitted is not None:
            return {
                "code_ids": set(submitted),
                "search_count": search_count,
                "iterations": iteration + 1,
                "input_tokens": total_input,
                "output_tokens": total_output,
                "log": "\n".join(log_lines)
            }

        messages.extend(tool_responses)
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

# ═══════════════════════════════════════════════════════════════════════════════
# AGENTIC LOOP — ANTHROPIC (Claude Sonnet 5)
# ═══════════════════════════════════════════════════════════════════════════════
def run_agentic_anthropic(model_id, system_prompt, dict_type):
    messages = [
        {"role": "user", "content": "Build the codelist."}
    ]

    total_input = 0
    total_output = 0
    search_count = 0
    nudge_count = 0
    log_lines = []

    for iteration in range(MAX_ITERATIONS):
        try:
            resp = anthropic_client.messages.create(
                model=model_id,
                system=system_prompt,
                max_tokens=16000,
                tools=ANTHROPIC_TOOLS,
                messages=messages
            )
        except Exception as e:
            log_lines.append(f"ERROR iter {iteration}: {e}")
            break

        total_input += resp.usage.input_tokens
        total_output += resp.usage.output_tokens

        messages.append({"role": "assistant", "content": resp.content})

        tool_uses = [b for b in resp.content if getattr(b, "type", None) == "tool_use"]

        if not tool_uses:
            if search_count > 0 and nudge_count < 2:
                log_lines.append(f"[iter {iteration}] No tool calls — nudging to submit.")
                messages.append({"role": "user",
                                 "content": "Now call submit_codelist with all the code IDs you want to include."})
                nudge_count += 1
                continue
            log_lines.append(f"[iter {iteration}] No tool calls, stopping.")
            break

        tool_results = []
        submitted = None

        for tu in tool_uses:
            if tu.name == "search_dictionary":
                query = tu.input.get("query", "")
                k = min(tu.input.get("k", K_DEFAULT), K_MAX)
                results = search_dictionary(query, dict_type, k)
                search_count += 1
                log_lines.append(f"[iter {iteration}] search({query!r}, k={k}) => {len(results)} results")
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tu.id,
                    "content": json.dumps(results)
                })

            elif tu.name == "submit_codelist":
                code_ids = tu.input.get("code_ids", [])
                log_lines.append(f"[iter {iteration}] submit_codelist({len(code_ids)} codes)")
                submitted = code_ids

        if submitted is not None:
            return {
                "code_ids": set(submitted),
                "search_count": search_count,
                "iterations": iteration + 1,
                "input_tokens": total_input,
                "output_tokens": total_output,
                "log": "\n".join(log_lines)
            }

        messages.append({"role": "user", "content": tool_results})

        if resp.stop_reason == "end_turn":
            if search_count > 0 and nudge_count < 2:
                log_lines.append(f"[iter {iteration}] end_turn — nudging to submit.")
                messages.append({"role": "user",
                                 "content": "Now call submit_codelist with all the code IDs you want to include."})
                nudge_count += 1
                continue
            log_lines.append(f"[iter {iteration}] end_turn without submit")
            break

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

# ═══════════════════════════════════════════════════════════════════════════════
# AGENTIC LOOP — GEMINI (native SDK, handles thought signatures)
# ═══════════════════════════════════════════════════════════════════════════════
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

    for iteration in range(MAX_ITERATIONS):
        try:
            resp = gemini_client.models.generate_content(
                model=model_id, contents=contents, config=config
            )
        except Exception as e:
            log_lines.append(f"ERROR iter {iteration}: {e}")
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

# ═══════════════════════════════════════════════════════════════════════════════
# UNIFIED RUNNER
# ═══════════════════════════════════════════════════════════════════════════════
def run_agentic(model_name, model_config, condition_name, dict_type):
    system_prompt = get_system_prompt(condition_name, dict_type)
    if model_config["client"] == "anthropic":
        return run_agentic_anthropic(model_config["model_id"], system_prompt, dict_type)
    elif model_config["client"] == "openai":
        return run_agentic_openai(openai_client, model_config["model_id"],
                                  system_prompt, dict_type)
    elif model_config["client"] == "gemini":
        return run_agentic_gemini(model_config["model_id"],
                                  system_prompt, dict_type)
    else:
        return run_agentic_openai(openai_client, model_config["model_id"],
                                  system_prompt, dict_type)

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
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("AGENTIC RAG PIPELINE")
print("Models:", ", ".join(MODELS.keys()))
print("=" * 80)

all_results = []
t_start = time.time()

for cond in CONDITIONS:
    ref_codes = cond["ref_loader"]()
    print(f"\n{'=' * 60}")
    print(f"  {cond['name']} | ref={len(ref_codes)} | dict={cond['dict_type']}")
    print(f"{'=' * 60}")

    for model_name, mcfg in MODELS.items():
        print(f"  [{model_name}] agentic loop...", end=" ", flush=True)
        t0 = time.time()

        try:
            result = run_agentic(model_name, mcfg, cond["name"], cond["dict_type"])
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
                "Condition": cond["name"], "Model": model_name,
                "N_searches": result["search_count"],
                "N_iterations": result["iterations"],
                "Input_tokens": result["input_tokens"],
                "Output_tokens": result["output_tokens"],
                **metrics
            }
            if "error" in result:
                row["Error"] = result["error"]
            all_results.append(row)

            tag = f"{cond['name']}_{model_name}".replace(" ", "_")
            pd.DataFrame({"code_id": sorted(ai_codes)}).to_csv(
                os.path.join(OUTPUT_DIR, f"{tag}_codelist.csv"), index=False)
            with open(os.path.join(OUTPUT_DIR, f"{tag}_log.txt"),
                      "w", encoding="utf-8") as f:
                f.write(result["log"])

        except Exception as e:
            dt = time.time() - t0
            print(f"ERROR ({dt:.1f}s): {str(e)[:150]}")
            all_results.append({
                "Condition": cond["name"], "Model": model_name,
                "Error": str(e)[:200]
            })

        time.sleep(1)

elapsed = time.time() - t_start
print(f"\nPipeline complete in {elapsed:.0f}s")

# ═══════════════════════════════════════════════════════════════════════════════
# SAVE & DISPLAY RESULTS
# ═══════════════════════════════════════════════════════════════════════════════
results_df = pd.DataFrame(all_results)
results_df.to_csv(os.path.join(OUTPUT_DIR, "evaluation_results_all.csv"), index=False)

print("\n" + "=" * 80)
print("RESULTS TABLE (Agentic RAG)")
print("=" * 80)
display_cols = ["Condition", "Model", "N_searches",
                "Ref_size", "AI_size", "Precision", "Recall", "F1", "Jaccard",
                "Input_tokens", "Output_tokens"]
valid = [c for c in display_cols if c in results_df.columns]
print(results_df[valid].to_string(index=False))

# Model averages
print("\n  Model averages:")
for m in MODELS:
    sub = results_df[results_df["Model"] == m]
    if "F1" in sub.columns and sub["F1"].notna().any():
        print(f"    {m:<20} mean F1={sub['F1'].mean():.3f}  "
              f"mean searches={sub['N_searches'].mean():.1f}  "
              f"total tokens={sub['Input_tokens'].sum():.0f}in/{sub['Output_tokens'].sum():.0f}out")

# ═══════════════════════════════════════════════════════════════════════════════
# AF INTER-EXPERT VARIATION
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("AF INTER-EXPERT VARIATION")
print("=" * 80)

af_experts = {}
for af in AF_EXPERT_LISTS:
    codes = af["loader"]()
    af_experts[af["name"]] = codes
    print(f"  {af['name']}: {len(codes)} codes")

names = list(af_experts.keys())
pairwise = []
for i in range(len(names)):
    for j in range(i + 1, len(names)):
        s1, s2 = af_experts[names[i]], af_experts[names[j]]
        inter = len(s1 & s2)
        union = len(s1 | s2)
        jacc = inter / union if union else 0
        pairwise.append({"Expert_1": names[i], "Expert_2": names[j],
                         "Jaccard": round(jacc, 4),
                         "Intersection": inter, "Union": union})

pd.DataFrame(pairwise).to_csv(
    os.path.join(OUTPUT_DIR, "af_expert_pairwise.csv"), index=False)
expert_jaccards = [p["Jaccard"] for p in pairwise]
print(f"  Expert Jaccard range: [{min(expert_jaccards):.4f}, {max(expert_jaccards):.4f}]")
print(f"  Expert Jaccard mean:  {np.mean(expert_jaccards):.4f}")

print("\nAI vs each AF expert:")
af_ai_results = []
for model_name in MODELS:
    tag = f"Atrial_Fibrillation_{model_name}".replace(" ", "_")
    ai_file = os.path.join(OUTPUT_DIR, f"{tag}_codelist.csv")
    if not os.path.exists(ai_file):
        continue
    ai_codes = set(pd.read_csv(ai_file, dtype=str)["code_id"].tolist())
    jaccards = []
    for en, ec in af_experts.items():
        m = evaluate(ai_codes, ec)
        jaccards.append(m["Jaccard"])
        af_ai_results.append({"Model": model_name, "Expert": en, **m})
    mean_j = np.mean(jaccards)
    flag = "WITHIN" if min(expert_jaccards) <= mean_j <= max(expert_jaccards) else "OUTSIDE"
    print(f"  {model_name}: mean Jaccard={mean_j:.4f} [{flag}]")

pd.DataFrame(af_ai_results).to_csv(
    os.path.join(OUTPUT_DIR, "af_ai_vs_experts.csv"), index=False)

# ═══════════════════════════════════════════════════════════════════════════════
# COMPARISON WITH PREVIOUS RESULTS
# ═══════════════════════════════════════════════════════════════════════════════
prev_file = os.path.join(BASE, "results_multiquery", "evaluation_results_all.csv")
if os.path.exists(prev_file):
    print("\n" + "=" * 80)
    print("COMPARISON: Agentic vs Multi-query Pipeline")
    print("=" * 80)
    prev = pd.read_csv(prev_file)
    for _, row in results_df.iterrows():
        if "F1" not in row or pd.isna(row.get("F1")):
            continue
        cond = row["Condition"]
        old = prev[prev["Condition"] == cond]
        if old.empty:
            continue
        best_old = old.loc[old["F1"].idxmax()]
        delta = row["F1"] - best_old["F1"]
        arrow = "+" if delta > 0 else ""
        print(f"  {cond:<30} {row['Model']:<20} F1={row['F1']:.3f}  "
              f"vs best pipeline ({best_old['Model']}) F1={best_old['F1']:.3f}  "
              f"({arrow}{delta:.3f})")

print("\n" + "=" * 80)
print("DONE")
print(f"  Results: {OUTPUT_DIR}")
print(f"  Time: {elapsed:.0f}s")
print("=" * 80)
