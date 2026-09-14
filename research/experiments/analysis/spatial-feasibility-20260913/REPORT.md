# Native spatial descriptor capability: partial outcome
Two bounded probes ended with distinct MPS errors. First: CLIP 14x14 to 4x4 adaptive average pooling is unsupported on this MPS version. CPU pooling fixed that operation. Second: all four CLIP pools (32 images) passed native/cached CLS parity, but DINO failed before feature inspection because the launch omitted the inherited PYTORCH_ENABLE_MPS_FALLBACK=1 setting.

This is an execution omission, not evidence against DINO features. Baseline gallery_recovery_manifest.json already records bicubic CPU fallback. No third standalone pilot will be run. DINO's existing forward_features interface exposes normalized native patch tokens; its full extraction must validate CLS and pooling as it writes each real batch. No classification accuracy was evaluated and no full DINO capability pass is claimed.

Exact identity inventory: DTD 1877 plus EuroSAT 5400 source query-pool images. Storing 16x384 float16 descriptors requires 89,419,776 bytes before tiny metadata. Free space was 10.574 GiB. Direct memmap output avoids duplicate chunk/final assets. The source gallery does not need new patches for a support-only local classifier.
