"""Step 1: Embed CPRD Aurum dictionary terms using OpenAI text-embedding-3-large.
Run once. Saves embeddings + metadata to disk for the RAG pipeline."""
import pandas as pd
import numpy as np
import os
import time
from openai import OpenAI

BASE = r"D:\Dissertation\file"
EMBED_MODEL = "text-embedding-3-large"
EMBED_DIM = 1536
BATCH_SIZE = 2048

client = OpenAI()


def embed_texts(texts, label=""):
    embeddings = []
    n_batches = (len(texts) + BATCH_SIZE - 1) // BATCH_SIZE
    t0 = time.time()
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]
        batch = [t if t.strip() else "unknown" for t in batch]
        for attempt in range(5):
            try:
                resp = client.embeddings.create(
                    model=EMBED_MODEL, input=batch, dimensions=EMBED_DIM
                )
                break
            except Exception as e:
                if "429" in str(e) or "rate" in str(e).lower():
                    wait = min(60, 2 ** attempt * 5)
                    print(f"  Rate limited, waiting {wait}s...")
                    time.sleep(wait)
                else:
                    raise
        embeddings.extend([e.embedding for e in resp.data])
        b = i // BATCH_SIZE + 1
        if b % 20 == 0 or b == n_batches:
            elapsed = time.time() - t0
            print(f"  [{label}] {b}/{n_batches} batches | "
                  f"{len(embeddings)}/{len(texts)} terms | {elapsed:.0f}s elapsed")
        time.sleep(0.5)
    return np.array(embeddings, dtype=np.float32)


# ── 1. Medical Dictionary ────────────────────────────────────────────────────
print("Loading CPRD Aurum Medical Dictionary...")
med = pd.read_csv(os.path.join(BASE, "CPRDAurumMedical.txt"), sep="\t",
                  dtype=str, low_memory=False)
med.columns = med.columns.str.strip()
med_terms = med["Term"].fillna("").str.strip().tolist()
print(f"  {len(med_terms)} terms loaded")

print(f"\nEmbedding {len(med_terms)} medical terms (~5 min)...")
med_emb = embed_texts(med_terms, "MED")
np.save(os.path.join(BASE, "med_embeddings.npy"), med_emb)
med[["MedCodeId", "Term"]].to_csv(os.path.join(BASE, "med_metadata.csv"), index=False)
print(f"  Saved: med_embeddings.npy {med_emb.shape}, med_metadata.csv")

del med_emb

# ── 2. Product Dictionary ────────────────────────────────────────────────────
print("\nLoading CPRD Aurum Product Dictionary...")
prod = pd.read_csv(os.path.join(BASE, "CPRDAurumProduct.txt"), sep="\t",
                   dtype=str, low_memory=False)
prod.columns = prod.columns.str.strip()
prod_text = (prod["ProductName"].fillna("") + " " +
             prod["DrugSubstanceName"].fillna("")).str.strip().tolist()
print(f"  {len(prod_text)} terms loaded")

print(f"\nEmbedding {len(prod_text)} product terms (~2 min)...")
prod_emb = embed_texts(prod_text, "PROD")
np.save(os.path.join(BASE, "prod_embeddings.npy"), prod_emb)
prod[["ProdCodeId", "ProductName"]].to_csv(
    os.path.join(BASE, "prod_metadata.csv"), index=False)
print(f"  Saved: prod_embeddings.npy {prod_emb.shape}, prod_metadata.csv")

print("\n=== All embeddings complete! ===")
