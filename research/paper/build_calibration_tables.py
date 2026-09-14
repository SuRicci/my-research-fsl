from pathlib import Path
import json,hashlib
P=Path(__file__).resolve().parent
B=P.parent/"experiments/main/prior-calibration-qualification-20260913"
a=json.loads((B/"outputs/analysis.json").read_text())
cells=a["cells"]; directions=a["directions"]
T=P/"latex/tables"; T.mkdir(exist_ok=True)
names=list(next(iter(cells.values()))["accuracy_pct"])
ordered=["eurosat_to_dtd_dtd_k1","eurosat_to_dtd_eurosat_k1","eurosat_to_dtd_dtd_k5","eurosat_to_dtd_eurosat_k5",
         "dtd_to_eurosat_eurosat_k1","dtd_to_eurosat_dtd_k1","dtd_to_eurosat_eurosat_k5","dtd_to_eurosat_dtd_k5"]
def label(n):
    if n=="r2": return "R2"
    if n.startswith("cs_"):return r"CS$_{\ell_2}$, $\lambda="+n[3:]+"$"
    if n.startswith("grid_"):
        _,mix,lam=n.split("_");return r"Mix $"+mix+r"$, $\lambda="+lam+"$"
    return {"logistic_1":"LR1","logistic_10":"LR10","calibrated":"Calibrated","same_mass":"Same mass",
            "composition_only":"Composition only","fixed_prior":"Fixed prior"}[n]
rows=[label(n)+" & "+" & ".join(f'{cells[c]["accuracy_pct"][n]:.2f}' for c in ordered)+r" \\" for n in names]
s=r"""\begin{table}[ht]
\centering
\caption{Complete development-transfer accuracy (\%). Each target uses the calibrator learned on the other domain. M and X denote matched and fully mismatched galleries. Each column contains 500 paired tasks. Five-shot fallback is specified in the text; duplicate rows reflect identical predictions rather than independent evidence.}\label{tab:calibration-accuracy}
\small\setlength{\tabcolsep}{3pt}
\begin{tabular}{lrrrrrrrr}\toprule
& \multicolumn{4}{c}{DTD} & \multicolumn{4}{c}{EuroSAT} \\
& \multicolumn{2}{c}{1-shot} & \multicolumn{2}{c}{5-shot}
& \multicolumn{2}{c}{1-shot} & \multicolumn{2}{c}{5-shot} \\
Method & M & X & M & X & M & X & M & X \\\midrule
"""+'\n'.join(rows)+r"""
\bottomrule\end{tabular}
\end{table}
"""
(T/"calibration_accuracy.tex").write_text(s)
rows=[]
for src,dest in [("dtd","EuroSAT"),("eurosat","DTD")]:
    c=next(v for k,v in cells.items() if k.startswith(src+"_to_"))
    winner=c["source_winner"];d=directions[src][winner]; lo,hi=d["ci95_pp"]
    rows.append(("DTD" if src=="dtd" else "EuroSAT")+r" $\to$ "+dest+" & "+label(winner)+f' & {d["delta_pp"]:+.2f} & [{lo:+.2f}, {hi:+.2f}] & 0.00'+r" \\")
s=r"""\begin{table}[ht]
\centering
\caption{One-shot paired accuracy differences in development transfer (percentage points). Each task is averaged across its two gallery conditions before resampling. The final column compares calibrated relevance with the same-mass control.}\label{tab:calibration-deltas}
\small\setlength{\tabcolsep}{4pt}
\begin{tabular}{llrrr}\toprule
Training $\to$ target & Source-selected control & $\Delta$ & 95\% interval & $\Delta_{\rm mass}$\\\midrule
"""+'\n'.join(rows)+r"""
\bottomrule\end{tabular}
\end{table}
"""
(T/"calibration_deltas.tex").write_text(s)
audit={"status":"generated","data_path":str(B/"outputs/analysis.json"),"data_sha256":hashlib.sha256((B/"outputs/analysis.json").read_bytes()).hexdigest(),"cell_count":len(cells),"method_count":len(names),"accuracy_values":len(cells)*len(names),"tables":["calibration_accuracy","calibration_deltas"]}
(P/"calibration_table_audit.json").write_text(json.dumps(audit,indent=2)+"\n")
print(json.dumps(audit))
