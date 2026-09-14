# Caltech101 prospective transfer asset qualification

Official package acquired and verified:137414764bytes,MD5 3138e1922a9193bfa496528edbbc45d0; all ZIP CRC checks passed. The archive contains8members. Images were streamed from the nested object tar; annotations and JPEG files were not extracted.

All9144images decoded successfully:8677object images in101classes and467background images. Background is excluded from classification. No exact RGB duplicate or conflicting-class group was found among object images. No overlap was found with33639unique decoded-RGB identities from8explicit current local manifests, including DTD,EuroSAT,Pets,CUB,Flowers andCIFAR. This is a bounded exact-image check; perceptual duplicates, encoder pretraining exposure and complete historical/remote nonexposure are not proved.

A prospective identity split was constructed by sorting per-class SHA256(seed26091393:rgb), first floor(n/3) gallery, remaining images support/query pool. It contains2859gallery images and5818query-pool images; all101classes remain, minimum31unique images and21query-pool images/class. Supports and queries will be drawn without replacement per episode; task identities and comparator protocol still must be locked before model outcomes. No Caltech feature or accuracy was computed.

Storage:11.109GiB free afterqualification. Full six-view float32 fused features for8677images require186590208bytes; archive137414764bytes plus one feature copy and at most one temporary feature copy,metadata andboundedlogs fit the0.6GiB acquisition ceiling if JPEG extraction and duplicate worktrees are avoided. Actual finalencodingstorage still requires measurement. Existing two encoders and MPS path are retained, not downloaded again.

Verdict:package,exactidentity,101-classpoolandboundedstorage gates pass. Next:prospective frozen two-source accumulated-stack validation, including cheaperparent,equalviewmeanR2/CS and singleviewR2, with source configurations gamma10/eta0 andgamma1/eta10 both required. Independent new-target outcomes will not rescue failed development gates or automatically validate a paper claim. Use separate Caltech metric ids; canonicalPetsmetricsunchanged.

Evidence:CONTRACT.json,acquire.py,acquisition.json,audit_identities.py,identities.json,identity_audit.json; download bash-e75b4039; RGB audit bash-e704b8cb; science-d8e8afbf. Source rights/history evidence remains in artifacts/idea/independent_assets/CALTECH101_QUALIFICATION.md.
