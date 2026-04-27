"""Compare JobA clean vs JobA val_defect across the 10 categories that were rerun."""
import json
import re
from pathlib import Path
from statistics import mean, median, stdev

ROOT = Path(__file__).resolve().parents[1] / "data" / "outputs"
MODELS = ["anomalib_patchcore", "anomalib_padim", "subspacead"]
CLEAN_RE = re.compile(r"^jobA_(?!val_defect_)(.+)_\d{8}_\d{6}$")
VD_RE = re.compile(r"^jobA_val_defect_(.+)_\d{8}_\d{6}$")


def _load(prefix_re):
    out = {}
    for d in sorted(ROOT.glob("jobA_*")):
        if "csflow" in d.name:
            continue
        m = prefix_re.match(d.name)
        if not m:
            continue
        cat = m.group(1)
        fp = d / "benchmark_summary.json"
        if not fp.exists():
            continue
        data = json.loads(fp.read_text())
        for entry in data.get("models", []):
            if entry["model"] not in MODELS:
                continue
            out[(cat, entry["model"])] = entry
    return out


clean = _load(CLEAN_RE)
vd = _load(VD_RE)
shared = sorted(set(clean) & set(vd))

if not shared:
    raise SystemExit("No paired (category, model) cells found.")

cats_in_vd = sorted({c for c, _ in vd})
print(f"val_defect categories: {len(cats_in_vd)} -> {', '.join(cats_in_vd)}")
print(f"paired (cat, model) cells: {len(shared)}\n")

# Detect whether a cell came from the legacy balanced-val experiment.
def _is_balanced_val_variant(entry, run_split):
    return ("recall_at_fpr_1pct" in entry) or ("val_balance" in run_split)


# Per-cell comparison table.
print("=== PER-CELL DELTAS ===")
header = ["category", "model", "v",
          "AUROC_c", "AUROC_v", "dAUROC",
          "AUPR_c", "AUPR_v", "dAUPR",
          "F1_c", "F1_v", "dF1",
          "Prec_c", "Prec_v", "dPrec",
          "Rec_c", "Rec_v", "dRec",
          "Thr_c", "Thr_v", "Thr_v/Thr_c",
          "valRec_v", "valF1_v"]
print("\t".join(header))

per_model_deltas = {m: {"dAUROC": [], "dAUPR": [], "dF1": [], "dPrec": [], "dRec": [],
                        "thr_ratio": [], "valF1": [], "valRec": []} for m in MODELS}

# Find run-level split metadata for each vd cell to detect the legacy
# balanced-val variant when mixed historical outputs are present.
def _split_for_vd(cat):
    for d in ROOT.glob(f"jobA_val_defect_{cat}_*"):
        fp = d / "benchmark_summary.json"
        if fp.exists():
            data = json.loads(fp.read_text())
            return data.get("dataset", {}).get("split", {})
    return {}


balanced_cells = 0
for cat, model in shared:
    c = clean[(cat, model)]
    v = vd[(cat, model)]
    is_balanced = _is_balanced_val_variant(v, _split_for_vd(cat))
    if is_balanced:
        balanced_cells += 1
    schema = "balanced" if is_balanced else "v1"

    auroc_c, auroc_v = c.get("auroc", 0), v.get("auroc", 0)
    aupr_c, aupr_v = c.get("aupr", 0), v.get("aupr", 0)
    f1_c, f1_v = c.get("f1", 0), v.get("f1", 0)
    p_c, p_v = c.get("precision", 0), v.get("precision", 0)
    r_c, r_v = c.get("recall", 0), v.get("recall", 0)
    th_c, th_v = c.get("threshold_used", 0), v.get("threshold_used", 0)
    thr_ratio = (th_v / th_c) if th_c else float("nan")

    per_model_deltas[model]["dAUROC"].append(auroc_v - auroc_c)
    per_model_deltas[model]["dAUPR"].append(aupr_v - aupr_c)
    per_model_deltas[model]["dF1"].append(f1_v - f1_c)
    per_model_deltas[model]["dPrec"].append(p_v - p_c)
    per_model_deltas[model]["dRec"].append(r_v - r_c)
    per_model_deltas[model]["thr_ratio"].append(thr_ratio)
    per_model_deltas[model]["valF1"].append(v.get("val_f1", 0))
    per_model_deltas[model]["valRec"].append(v.get("val_recall", 0))

    row = [
        cat, model.replace("anomalib_", ""), schema,
        f"{auroc_c:.3f}", f"{auroc_v:.3f}", f"{auroc_v - auroc_c:+.3f}",
        f"{aupr_c:.3f}", f"{aupr_v:.3f}", f"{aupr_v - aupr_c:+.3f}",
        f"{f1_c:.3f}", f"{f1_v:.3f}", f"{f1_v - f1_c:+.3f}",
        f"{p_c:.3f}", f"{p_v:.3f}", f"{p_v - p_c:+.3f}",
        f"{r_c:.3f}", f"{r_v:.3f}", f"{r_v - r_c:+.3f}",
        f"{th_c:.2f}", f"{th_v:.2f}", f"{thr_ratio:.2f}",
        f"{v.get('val_recall', 0):.3f}", f"{v.get('val_f1', 0):.3f}",
    ]
    print("\t".join(row))

print(
    f"\nschema mix: balanced-val legacy cells = {balanced_cells} / {len(shared)} "
    "(rest are v1: patched splitter + val_f1 only)"
)

print("\n=== PER-MODEL SUMMARY (mean delta over paired cells) ===")
print("model\tn\tdAUROC\tdAUPR\tdF1\tdPrec\tdRec\tthr_ratio\tvalRec\tvalF1")
for m in MODELS:
    d = per_model_deltas[m]
    n = len(d["dAUROC"])
    if n == 0:
        continue
    print(f"{m.replace('anomalib_','')}\t{n}"
          f"\t{mean(d['dAUROC']):+.4f}"
          f"\t{mean(d['dAUPR']):+.4f}"
          f"\t{mean(d['dF1']):+.4f}"
          f"\t{mean(d['dPrec']):+.4f}"
          f"\t{mean(d['dRec']):+.4f}"
          f"\t{mean(d['thr_ratio']):.3f}"
          f"\t{mean(d['valRec']):.3f}"
          f"\t{mean(d['valF1']):.3f}")

print("\n=== PER-MODEL SUMMARY (median delta) ===")
print("model\tdAUROC\tdAUPR\tdF1\tdPrec\tdRec\tthr_ratio")
for m in MODELS:
    d = per_model_deltas[m]
    n = len(d["dAUROC"])
    if n == 0:
        continue
    print(f"{m.replace('anomalib_','')}"
          f"\t{median(d['dAUROC']):+.4f}"
          f"\t{median(d['dAUPR']):+.4f}"
          f"\t{median(d['dF1']):+.4f}"
          f"\t{median(d['dPrec']):+.4f}"
          f"\t{median(d['dRec']):+.4f}"
          f"\t{median(d['thr_ratio']):.3f}")

# Sanity gates from PLAN.
print("\n=== SANITY GATES ===")
all_dauroc = []
all_daupr = []
val_recall_zero = 0
for m in MODELS:
    all_dauroc.extend(per_model_deltas[m]["dAUROC"])
    all_daupr.extend(per_model_deltas[m]["dAUPR"])
    val_recall_zero += sum(1 for x in per_model_deltas[m]["valRec"] if x == 0)

print(f"|median dAUROC| < 0.01 ?  median = {median(all_dauroc):+.4f}  -> {'PASS' if abs(median(all_dauroc)) < 0.01 else 'CHECK'}")
print(f"|median dAUPR|  < 0.01 ?  median = {median(all_daupr):+.4f}  -> {'PASS' if abs(median(all_daupr)) < 0.01 else 'CHECK'}")
print(f"val_recall == 0 cells:    {val_recall_zero}  -> {'PASS' if val_recall_zero == 0 else 'CHECK'}")

# Recall@FPR breakout for cells that include the industrial metrics fields.
print("\n=== INDUSTRIAL METRICS (cells with recall_at_fpr_* fields) ===")
print("category\tmodel\tR@FPR1\tR@FPR5\tmacro_rec\tweighted_rec")
for cat, model in shared:
    v = vd[(cat, model)]
    if "recall_at_fpr_1pct" not in v:
        continue
    print(f"{cat}\t{model.replace('anomalib_','')}"
          f"\t{v.get('recall_at_fpr_1pct', 0):.3f}"
          f"\t{v.get('recall_at_fpr_5pct', 0):.3f}"
          f"\t{v.get('macro_recall', 0):.3f}"
          f"\t{v.get('weighted_recall', 0):.3f}")

# Train / val / test sizes for cells that include the industrial metrics fields.
print("\n=== TRAIN/VAL/TEST SIZES (cells with recall_at_fpr_* fields) ===")
print("category\tmodel\ttrain\tval\ttest")
for cat, model in shared:
    v = vd[(cat, model)]
    if "recall_at_fpr_1pct" not in v:
        continue
    print(f"{cat}\t{model.replace('anomalib_','')}\t{v.get('train_samples')}\t{v.get('val_samples')}\t{v.get('test_samples')}")
