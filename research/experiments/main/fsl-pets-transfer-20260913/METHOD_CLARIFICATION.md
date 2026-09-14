# Source-level comparator clarification before target outcomes

CS_l2 refers to support-origin centering and L2 normalization applied to supports, each query independently, and gallery vectors. In one-shot tasks it then retrieves64 gallery neighbors, blends the normalized mean with the class prototype at0.5, and fits ridge0.1. It is a gallery method, not a support-only classifier. Pure support controls are logistic C1/C10.

For this new four-cell scope, the prespecified five-shot CS_l2 control applies the same support-origin geometry but uses only real supports and ridge0.1; its behavior is an explicit fixed extension, not an inherited result. R2 five-shot remains raw-feature support ridge1. All fields are in protocol.json. No Pets prediction was computed before this clarification; acquisition was still downloading. Source function prepare/head is reused verbatim.

The actual ilpcz.py is the formula-based local iLPC-z from the2025 Pattern Recognition extension, not loss-cleaning ICCV2021 exactly. Prior SOURCE_AUDIT.md records equations and unavailable source entrypoint. It has no query/gallery labels in adaptation, max3 pseudo-examples per class/round, C10 final head and explicit exhausted-class stop. Preserve all these caveats.
