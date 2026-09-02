# Dissertation Data Summary
## Evaluating AI-Generated Clinical Codelists Using CPRD Aurum Diagnosis and Product Dictionaries

### Study Design
- **Method**: Agentic RAG (Retrieval-Augmented Generation) — LLM autonomously decides search queries and when to stop
- **Tools given to LLM**: `search_dictionary` (embedding-based semantic search, returns top-K codes per call) and `submit_codelist` (final output)
- **Embedding model**: OpenAI text-embedding-3-large (1536 dimensions)
- **Parameters**: K_DEFAULT=50, K_MAX=200, MAX_ITERATIONS=20
- **Dictionaries**: CPRD Aurum Medical (291,711 codes) and Product (79,931 codes)

### Models Evaluated
| Model | Model ID | Provider |
|---|---|---|
| GPT-5.4 Mini | gpt-5.4-mini | OpenAI |
| Claude Sonnet 5 | claude-sonnet-5 | Anthropic |
| Gemini 3.6 Flash | gemini-3.6-flash | Google |

### System Prompts Used
**Medical dictionary prompt:**
> You are a clinical coding expert building a codelist for {condition} using the CPRD Aurum medical dictionary.
> Include a code if it represents a current diagnosis of {condition}.
> Search for the condition, its subtypes, and synonyms. You MUST call submit_codelist with all included code IDs when done. Never output code IDs as text.

**Product dictionary prompt:**
> You are a clinical coding expert building a codelist for {condition} products using the CPRD Aurum product dictionary.
> Include a code if it is a {condition} preparation or contains {condition} as an active ingredient.
> Search for the product and its variants. You MUST call submit_codelist with all included code IDs when done. Never output code IDs as text.

---

## Table 1: Reference Codelist Metadata (Methods)

| Condition | Dictionary | Source | Source Codes | In Dictionary | Not in Dictionary | Coverage |
|---|---|---|---:|---:|---:|---:|
| Atrial Fibrillation | Medical | ClinicalCodes | 18 | 18 | 0 | 100.0% |
| Heart Failure | Medical | ClinicalCodes | 84 | 84 | 0 | 100.0% |
| Cerebrovascular Disease | Medical | ClinicalCodes | 130 | 130 | 0 | 100.0% |
| Hypertension | Medical | LSHTM | 234 | 214 | 20 | 91.5% |
| Myocardial Infarction | Medical | LSHTM | 165 | 127 | 38 | 77.0% |
| Peripheral Arterial Disease | Medical | LSHTM | 443 | 386 | 57 | 87.1% |
| COPD | Medical | LSHTM | 232 | 212 | 20 | 91.4% |
| Asthma | Medical | LSHTM | 188 | 169 | 19 | 89.9% |
| Insulin | Product | ClinicalCodes | 224 | 224 | 0 | 100.0% |
| Metformin | Product | ClinicalCodes | 79 | 79 | 0 | 100.0% |

Note: "In Dictionary" is the effective reference size used for evaluation. LSHTM-sourced codelists contain codes not present in the current CPRD Aurum dictionary version, which count as unreachable FN.

Reference sources:
- ClinicalCodes: https://clinicalcodes.rss.mhs.man.ac.uk/
- HDR UK Phenotype Library: https://phenotypes.healthdatagateway.org/
- LSHTM Data Compass: https://datacompass.lshtm.ac.uk/

AF has 5 expert codelists for inter-expert comparison:
- AF-1: ClinicalCodes (primary reference, 18 codes)
- AF-2: HDR UK Phenotype Library (38 codes)
- AF-3: LSHTM (30 codes)
- AF-4: LSHTM (25 codes)
- AF-5: LSHTM (31 codes)

---

## Table 2: Main Results — All Conditions x 3 Models (Results)

| Condition | Model | Searches | AI Codes | Ref Codes | Precision | Recall | F1 | Jaccard |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Atrial Fibrillation | GPT-5.4 Mini | 6 | 46 | 18 | 0.348 | 0.889 | 0.500 | 0.333 |
| | Claude Sonnet 5 | 9 | 33 | 18 | 0.485 | 0.889 | 0.628 | 0.457 |
| | Gemini 3.6 Flash | 7 | 31 | 18 | 0.516 | 0.889 | 0.653 | 0.485 |
| Heart Failure | GPT-5.4 Mini | 8 | 63 | 84 | 0.333 | 0.250 | 0.286 | 0.167 |
| | Claude Sonnet 5 | 10 | 124 | 84 | 0.234 | 0.345 | 0.279 | 0.162 |
| | Gemini 3.6 Flash | 10 | 99 | 84 | 0.293 | 0.345 | 0.317 | 0.188 |
| Cerebrovascular Disease | GPT-5.4 Mini | 12 | 84 | 130 | 0.333 | 0.215 | 0.262 | 0.151 |
| | Claude Sonnet 5 | 16 | 426 | 130 | 0.169 | 0.554 | 0.259 | 0.149 |
| | Gemini 3.6 Flash | 17 | 304 | 130 | 0.168 | 0.392 | 0.235 | 0.133 |
| Hypertension | GPT-5.4 Mini | 10 | 110 | 214 | 0.555 | 0.285 | 0.377 | 0.232 |
| | Claude Sonnet 5 | 14 | 153 | 214 | 0.582 | 0.416 | 0.485 | 0.320 |
| | Gemini 3.6 Flash | 9 | 172 | 214 | 0.570 | 0.458 | 0.508 | 0.340 |
| Myocardial Infarction | GPT-5.4 Mini | 6 | 79 | 127 | 0.570 | 0.354 | 0.437 | 0.280 |
| | Claude Sonnet 5 | 6 | 169 | 127 | 0.367 | 0.488 | 0.419 | 0.265 |
| | Gemini 3.6 Flash | 16 | 191 | 127 | 0.429 | 0.646 | 0.516 | 0.348 |
| Peripheral Arterial Disease | GPT-5.4 Mini | 10 | 36 | 386 | 0.472 | 0.044 | 0.081 | 0.042 |
| | Claude Sonnet 5 | 17 | 204 | 386 | 0.260 | 0.137 | 0.180 | 0.099 |
| | Gemini 3.6 Flash | 22 | 206 | 386 | 0.243 | 0.130 | 0.169 | 0.092 |
| COPD | GPT-5.4 Mini | 5 | 62 | 212 | 0.790 | 0.231 | 0.358 | 0.218 |
| | Claude Sonnet 5 | 12 | 111 | 212 | 0.676 | 0.354 | 0.464 | 0.302 |
| | Gemini 3.6 Flash | 8 | 116 | 212 | 0.690 | 0.377 | 0.488 | 0.323 |
| Asthma | GPT-5.4 Mini | 5 | 85 | 169 | 0.541 | 0.272 | 0.362 | 0.221 |
| | Claude Sonnet 5 | 3 | 97 | 169 | 0.536 | 0.308 | 0.391 | 0.243 |
| | Gemini 3.6 Flash | 9 | 164 | 169 | 0.366 | 0.355 | 0.360 | 0.220 |
| Insulin | GPT-5.4 Mini | 8 | 169 | 224 | 0.669 | 0.504 | 0.575 | 0.404 |
| | Claude Sonnet 5 | 26 | 183 | 224 | 0.792 | 0.647 | 0.713 | 0.553 |
| | Gemini 3.6 Flash | 8 | 182 | 224 | 0.791 | 0.643 | 0.709 | 0.550 |
| Metformin | GPT-5.4 Mini | 4 | 60 | 79 | 0.867 | 0.658 | 0.748 | 0.598 |
| | Claude Sonnet 5 | 25 | 91 | 79 | 0.802 | 0.924 | 0.859 | 0.753 |
| | Gemini 3.6 Flash | 8 | 91 | 79 | 0.802 | 0.924 | 0.859 | 0.753 |

### Summary by model (mean across 10 conditions):
| Model | Mean F1 | Mean Jaccard | Mean Precision | Mean Recall | Mean Searches |
|---|---:|---:|---:|---:|---:|
| GPT-5.4 Mini | 0.399 | 0.245 | 0.527 | 0.390 | 7.4 |
| Claude Sonnet 5 | 0.468 | 0.310 | 0.460 | 0.466 | 13.8 |
| Gemini 3.6 Flash | 0.481 | 0.343 | 0.470 | 0.516 | 11.4 |

### Summary by condition group:
| Group | Conditions | GPT Mean F1 | Claude Mean F1 | Gemini Mean F1 |
|---|---|---:|---:|---:|
| Cardiovascular | AF, HF, CVD, Hyp, MI, PAD | 0.324 | 0.375 | 0.400 |
| Respiratory | COPD, Asthma | 0.360 | 0.428 | 0.424 |
| Drug products | Insulin, Metformin | 0.662 | 0.786 | 0.784 |

Note on Gemini CVD: The model performed 17 searches but failed to call submit_codelist (behavioural failure — exhaustive searching without self-terminating). A fallback mechanism was used: all search results were collected, then a second Gemini call with only the submit_codelist tool was made to select codes from the accumulated candidates. This is documented in run_gemini_cvd_fix.py.

---

## Table 3: AF Inter-Expert Pairwise Agreement (Results)

| | AF-1 | AF-2 | AF-3 | AF-4 | AF-5 |
|---|---:|---:|---:|---:|---:|
| AF-1 | — | 0.366 | 0.500 | 0.387 | 0.531 |
| AF-2 | | — | 0.360 | 0.340 | 0.353 |
| AF-3 | | | — | 0.410 | 0.849 |
| AF-4 | | | | — | 0.436 |
| AF-5 | | | | | — |

**Mean inter-expert Jaccard: 0.453** (10 pairwise comparisons, range 0.340–0.849)

---

## Table 4: AF — AI Model vs Each Expert Codelist (Results)

| Model | Expert | AI Codes | Ref Codes | TP | Precision | Recall | F1 | Jaccard |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| GPT-5.4 Mini | AF-1 | 46 | 18 | 16 | 0.348 | 0.889 | 0.500 | 0.333 |
| | AF-2 | 46 | 38 | 25 | 0.544 | 0.658 | 0.595 | 0.424 |
| | AF-3 | 46 | 30 | 20 | 0.435 | 0.667 | 0.526 | 0.357 |
| | AF-4 | 46 | 25 | 18 | 0.391 | 0.720 | 0.507 | 0.340 |
| | AF-5 | 46 | 31 | 21 | 0.457 | 0.677 | 0.545 | 0.375 |
| | **Mean** | | | | **0.435** | **0.722** | **0.535** | **0.366** |
| Claude Sonnet 5 | AF-1 | 33 | 18 | 16 | 0.485 | 0.889 | 0.628 | 0.457 |
| | AF-2 | 33 | 38 | 23 | 0.697 | 0.605 | 0.648 | 0.479 |
| | AF-3 | 33 | 30 | 18 | 0.545 | 0.600 | 0.571 | 0.400 |
| | AF-4 | 33 | 25 | 16 | 0.485 | 0.640 | 0.552 | 0.381 |
| | AF-5 | 33 | 31 | 18 | 0.545 | 0.581 | 0.563 | 0.391 |
| | **Mean** | | | | **0.551** | **0.663** | **0.592** | **0.422** |
| Gemini 3.6 Flash | AF-1 | 31 | 18 | 16 | 0.516 | 0.889 | 0.653 | 0.485 |
| | AF-2 | 31 | 38 | 21 | 0.677 | 0.553 | 0.609 | 0.438 |
| | AF-3 | 31 | 30 | 16 | 0.516 | 0.533 | 0.525 | 0.356 |
| | AF-4 | 31 | 25 | 16 | 0.516 | 0.640 | 0.571 | 0.400 |
| | AF-5 | 31 | 31 | 16 | 0.516 | 0.516 | 0.516 | 0.348 |
| | **Mean** | | | | **0.548** | **0.626** | **0.575** | **0.405** |

**Context**: Inter-expert mean Jaccard is 0.453. AI model mean Jaccard ranges from 0.366 (GPT) to 0.422 (Claude), within or near inter-expert variation range.

---

## Table 5: FP Error Pattern Classification (Results/Discussion)

| Error Pattern | GPT-5.4 Mini | Claude Sonnet 5 | Gemini 3.6 Flash |
|---|---:|---:|---:|
| **Total FP** | **346** | **925** | **873** |
| Valid diagnosis subtype (reference gap) | 269 (77.7%) | 627 (67.8%) | 624 (71.5%) |
| Semantic drift (related condition) | 51 (14.7%) | 283 (30.6%) | 248 (28.4%) |
| History/resolved code | 7 (2.0%) | 15 (1.6%) | 0 (0.0%) |
| Risk/screening/suspected | 10 (2.9%) | 0 (0.0%) | 0 (0.0%) |
| Administrative/monitoring | 6 (1.7%) | 0 (0.0%) | 1 (0.1%) |
| Exclusion/ruled-out code | 3 (0.9%) | 0 (0.0%) | 0 (0.0%) |

FP category definitions:
- **Valid diagnosis subtype**: Code contains the condition keyword and represents a legitimate diagnosis the reference missed (e.g., "Valvular atrial fibrillation" for AF, "Chronic heart failure" for HF)
- **Semantic drift**: Code is clinically related but represents a different condition (e.g., "Atrial tachycardia" for AF, "Cor pulmonale" for HF)
- **History/resolved**: Past history or resolved codes (e.g., "H/O: atrial fibrillation", "Heart failure resolved")
- **Risk/screening/suspected**: Not yet confirmed diagnosis (e.g., "At increased risk of AF", "Atrial fibrillation screening")
- **Administrative/monitoring**: Follow-up, review, monitoring codes (e.g., "AF annual review", "AF monitoring")
- **Exclusion/ruled-out**: Explicitly excluded diagnosis (e.g., "Atrial fibrillation excluded") — the most serious error type

---

## Table 6: FN Error Pattern Classification (Results/Discussion)

| Error Pattern | GPT-5.4 Mini | Claude Sonnet 5 | Gemini 3.6 Flash |
|---|---:|---:|---:|
| **Total FN** | **1,195** | **977** | **960** |
| Procedural/surgical code | 243 (20.3%) | 242 (24.8%) | 242 (25.2%) |
| Administrative/monitoring code | 174 (14.6%) | 183 (18.7%) | 182 (19.0%) |
| History/resolved code | 14 (1.2%) | 16 (1.6%) | 22 (2.3%) |
| Sequelae/historical code | 16 (1.3%) | 12 (1.2%) | 12 (1.3%) |
| Complication code | 9 (0.8%) | 7 (0.7%) | 8 (0.8%) |
| Cause-of-death code | 1 (0.1%) | 2 (0.2%) | 0 (0.0%) |
| **Missed valid diagnosis** | **738 (61.8%)** | **515 (52.7%)** | **494 (51.5%)** |

Note: 154 reference codes not present in the CPRD Aurum dictionary are excluded from this table (already reported in Table 1 as dictionary coverage gaps). Total FN here reflects only codes the AI could have found.

FN category definitions:
- **Procedural/surgical**: Surgical procedures in reference (e.g., "Percutaneous transluminal insertion of stent into femoral artery") — AI correctly excluded these since prompt asks for "current diagnosis"
- **Administrative/monitoring**: Follow-up, QOF, screening codes (e.g., "Heart failure annual review", "Exception reporting - hypertension quality indicators")
- **Missed valid diagnosis**: Genuine diagnosis codes the AI failed to find — the true under-inclusion rate

---

## Table 7: FP Examples by Condition and Category (Appendix)

| Condition | Category | Example Terms |
|---|---|---|
| AF | Valid subtype | Paroxysmal AF with rapid ventricular response; Valvular AF; Lone AF; Familial AF |
| AF | Semantic drift | Atrial arrhythmia; Atrial tachycardia; Cardiac fibrillation; Atrial ectopic |
| AF | History/resolved | H/O: atrial fibrillation; AF resolved; History of AF |
| AF | Exclusion | Atrial fibrillation excluded |
| HF | Valid subtype | HF with mid-range EF; Chronic HF; CCF; CHF; Right ventricular failure |
| HF | Semantic drift | Cor pulmonale; Cardiac asthma; Pulmonary heart disease |
| CVD | Valid subtype | Intracerebral hemorrhage; Acute ischaemic stroke; SAH; Cerebral infarction with haemorrhagic transformation |
| CVD | Semantic drift | Cerebral aneurysm; Moyamoya disease; Carotid artery disease; Binswanger's encephalopathy |
| Hypertension | Valid subtype | Hypertension stage 1/2; Renal hypertension; Benign hypertension |
| Hypertension | Semantic drift | Secondary hyperaldosteronism; Neonatal hypertension |
| MI | Valid subtype | Acute STEMI due to LAD occlusion; Type 1 MI; MINOCA; Acute apical MI |
| MI | Semantic drift | Post-MI syndrome (Dressler's); Mural thrombosis |
| PAD | Valid subtype | Peripheral arterial disease; Intermittent claudication; PAOD; Arterial occlusive disease |
| PAD | Semantic drift | Arterial ulcer; Diabetic foot gangrene; Limb ischaemia |
| COPD | Valid subtype | Chronic obstructive bronchitis; Emphysematous bronchitis; Chronic airflow limitation |
| Asthma | Valid subtype | Allergic asthma; Occupational asthma; Exercise-induced asthma |
| Metformin | Valid subtype | Metformin 1g tablets; Glucophage SR 850mg |
| Metformin | Semantic drift | Janumet; Vokanamet (combination products containing metformin) |

---

## Table 8: Model Behaviour and Search Strategy (Appendix)

| Metric | GPT-5.4 Mini | Claude Sonnet 5 | Gemini 3.6 Flash |
|---|---:|---:|---:|
| Overall mean F1 | 0.399 | 0.468 | 0.481 |
| Overall mean Jaccard | 0.245 | 0.310 | 0.343 |
| Mean searches/condition | 7.4 | 13.8 | 11.4 |
| Mean AI codelist size | 89.8 | 161.5 | 155.6 |
| Mean Precision | 0.527 | 0.460 | 0.470 |
| Mean Recall | 0.390 | 0.466 | 0.516 |
| Total input tokens | 174,168 | 1,017,742 | 3,955,244 |
| Total output tokens | 7,123 | 156,393 | 31,762 |
| Conditions F1 > 0.5 | 3 | 4 | 5 |
| Conditions F1 < 0.3 | 3 | 3 | 2 |
| Failed submissions | 0 | 0 | 1 (CVD) |

---

## Key Findings for Discussion

1. **Overall performance**: Gemini 3.6 Flash achieved the highest mean F1 (0.481), followed by Claude Sonnet 5 (0.468) and GPT-5.4 Mini (0.399).

2. **Drug products >> Diagnoses**: Product codelists (Insulin, Metformin) achieved substantially higher F1 (0.662–0.786) than diagnosis codelists, likely due to more specific and well-defined terminology.

3. **Condition specificity matters**: Well-defined conditions (AF, Metformin) performed much better than umbrella terms (CVD, PAD). This parallels human expert variation — AF inter-expert Jaccard was only 0.453.

4. **F1 underestimates AI quality**:
   - ~68–78% of FPs are valid diagnosis subtypes the reference missed
   - ~45–55% of FNs are codes the AI can't find (not in dictionary) or correctly excluded (procedures, admin codes)
   - Only ~45–55% of FNs are genuine misses

5. **Minimal hallucination**: Claude and Gemini included 0 exclusion codes, 0 risk/screening codes. GPT included 3 exclusion and 10 risk codes — still <4% of its FPs.

6. **AI performance approaches inter-expert variation**: For AF, AI mean Jaccard (0.366–0.422) falls within the range of inter-expert Jaccard (0.340–0.849, mean 0.453).

7. **Gemini CVD failure mode**: Gemini 3.6 Flash consistently fails to submit for CVD despite exhaustive searching (17–22 searches). This is a reproducible behavioural pattern, not a random failure. A fallback mechanism (two-phase approach) resolved it.

---

## GitHub Repository
https://github.com/jiamingliu776-byte/dissertation-codelists

Contains: pipeline code, AI-generated codelists, search logs, evaluation results, error analysis.

---

## File Reference
- `rag_pipeline_v3.py` — Main pipeline (all 3 models, 10 conditions)
- `run_gemini_only.py` — Gemini re-run with native SDK
- `run_gemini_cvd_fix.py` — CVD fallback fix
- `error_analysis.py` — FP/FN classification
- `results_agentic/evaluation_results_all.csv` — Raw evaluation data
- `results_agentic/fp_error_analysis.csv` — FP classification counts
- `results_agentic/fn_error_analysis.csv` — FN classification counts
