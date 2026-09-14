# Canonical comparator variant
Objective: internally comparable fresh embeddings for DTD and EuroSAT, original image splits and RGB dedup, canonical OpenAI CLIP B16 QuickGELU and native DINOv2 S14.
Required because historical image/cache parity failed; do not overwrite legacy cached comparator.
Success: query/gallery same transforms and model, disjoint RGB identities, finite and normalized cached rows, required R2 metrics before candidates.
Risks: limited2-domain panel and historical exposure; MPS32 vs CUDA16 is not historical parity.
No encoder training, no gallery labels during adaptation, no query-batch optimization.
Monitor: managed encode_canonical.py run and assets/manifest.json.
