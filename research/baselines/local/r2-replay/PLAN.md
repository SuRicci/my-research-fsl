# Comparator asset extension
Purpose: validate image/cache identities and recover missing DTD/EuroSAT galleries using the original splits and encoder weights.
No candidate changes. Current accepted metric remains DTD5.
Success: image hashes disjoint, sample cosine >=0.999 before gallery encoding, finite cached rows and reproducible ids.
Downgrade: if sample parity fails stop and explain before claiming the mixed cache comparable. Current MPS32 vs historical CUDA16 difference is retained.
Monitor: bash managed recover_gallery process; assets/gallery_recovery_manifest.json and saved chunks.
