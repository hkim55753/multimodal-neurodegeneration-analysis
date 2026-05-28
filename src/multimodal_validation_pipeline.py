import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc

# ============================================================
# CONFIG
# ============================================================

DATA_DIR = "data"
FIG_DIR = "figures"

os.makedirs(FIG_DIR, exist_ok=True)

# ============================================================
# LOAD DATA
# ============================================================

print("Loading datasets...")

demo = pd.read_csv(os.path.join(DATA_DIR, "Demographics_28May2026.csv"), low_memory=False)
cohort = pd.read_csv(os.path.join(DATA_DIR, "Subject_Cohort_History_28May2026.csv"), low_memory=False)
updrs = pd.read_csv(os.path.join(DATA_DIR, "MDS-UPDRS_Part_III_28May2026.csv"), low_memory=False)
moca = pd.read_csv(os.path.join(DATA_DIR, "Montreal_Cognitive_Assessment__MoCA__28May2026.csv"), low_memory=False)
gds = pd.read_csv(os.path.join(DATA_DIR, "Geriatric_Depression_Scale__Short_Version__28May2026.csv"), low_memory=False)
datscan = pd.read_csv(os.path.join(DATA_DIR, "DaTScan_SBR_Analysis_28May2026.csv"), low_memory=False)

print("Datasets loaded.")

# ============================================================
# FIRST VISIT
# ============================================================

def first_visit(df):
    if "PATNO" not in df.columns:
        return df
    return df.sort_values("PATNO").groupby("PATNO").first().reset_index()

demo = first_visit(demo)
cohort = first_visit(cohort)
updrs = first_visit(updrs)
moca = first_visit(moca)
gds = first_visit(gds)
datscan = first_visit(datscan)

# ============================================================
# COLUMN REDUCTION (UNCHANGED)
# ============================================================

demo = demo[["PATNO"] + [c for c in ["AGE", "AGE_AT_VISIT", "SEX"] if c in demo.columns]]
cohort = cohort[["PATNO"] + [c for c in ["COHORT"] if c in cohort.columns]]
updrs = updrs[["PATNO", "NP3TOT"]]
moca = moca[["PATNO", "MCATOT"]]

gds_items = [
    "GDSSATIS","GDSDROPD","GDSEMPTY","GDSBORED","GDSGSPIR",
    "GDSAFRAD","GDSHAPPY","GDSHLPLS","GDSHOME","GDSMEMRY",
    "GDSALIVE","GDSWRTLS","GDSENRGY","GDSHOPLS","GDSBETER"
]

gds[gds_items] = gds[gds_items].apply(pd.to_numeric, errors="coerce")
gds["GDSSCORE"] = gds[gds_items].sum(axis=1)
gds = gds[["PATNO", "GDSSCORE"]]

sbr_cols = [
    "DATSCAN_CAUDATE_R",
    "DATSCAN_CAUDATE_L",
    "DATSCAN_PUTAMEN_R",
    "DATSCAN_PUTAMEN_L"
]

available = [c for c in sbr_cols if c in datscan.columns]

for c in available:
    datscan[c] = pd.to_numeric(datscan[c], errors="coerce")

datscan["MEAN_STRIATAL_SBR"] = datscan[available].mean(axis=1)
datscan = datscan[["PATNO", "MEAN_STRIATAL_SBR"]]

# ============================================================
# MERGE
# ============================================================

print("\nMerging datasets...")

dfs = [demo, cohort, updrs, moca, gds, datscan]

merged = dfs[0]
for df in dfs[1:]:
    merged = pd.merge(merged, df, on="PATNO", how="inner")

merged = merged.dropna()

merged["COGNITIVE_IMPAIRMENT"] = (merged["MCATOT"] < 26).astype(int)

# ============================================================
# 🔥 FIX 1: CORRELATION HEATMAP (CUT-OFF TEXT FIXED)
# ============================================================

plt.figure(figsize=(9, 7))

sns.heatmap(
    merged[["NP3TOT","MCATOT","GDSSCORE","MEAN_STRIATAL_SBR"]].corr(),
    annot=True,
    cmap="coolwarm",
    linewidths=0.5
)

plt.title("Correlation Matrix")

# IMPORTANT FIX: prevents cutoff labels
plt.tight_layout()
plt.subplots_adjust(left=0.2, bottom=0.2)

plt.savefig(
    os.path.join(FIG_DIR, "correlation_heatmap.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Saved corrected heatmap")

# ============================================================
# SCATTER (UNCHANGED)
# ============================================================

plt.figure(figsize=(8, 6))

plt.scatter(
    merged["MEAN_STRIATAL_SBR"],
    merged["MCATOT"],
    alpha=0.7
)

plt.xlabel("Mean Striatal Binding Ratio")
plt.ylabel("MoCA Total Score")
plt.title("DAT Binding vs Cognitive Performance")

plt.tight_layout()

plt.savefig(
    os.path.join(FIG_DIR, "moca_vs_sbr.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ============================================================
# 🔥 FIX 2: STRATIFICATION PLOT (MISSING AXIS LABELS FIXED)
# ============================================================

plt.figure(figsize=(8, 6))

colors = merged["COGNITIVE_IMPAIRMENT"].map({0: "blue", 1: "red"})

plt.scatter(
    merged["MEAN_STRIATAL_SBR"],
    merged["MCATOT"],
    c=colors,
    alpha=0.6
)

# ✅ FIXED AXIS LABELS (WERE MISSING BEFORE)
plt.xlabel("Mean Striatal Binding Ratio (DAT SPECT)")
plt.ylabel("MoCA Total Score")

plt.title("Cognitive Impairment Stratification")

import matplotlib.patches as mpatches

legend_elements = [
    mpatches.Patch(color='blue', label='No impairment'),
    mpatches.Patch(color='red', label='Impairment')
]

plt.legend(handles=legend_elements)

plt.tight_layout()

plt.savefig(
    os.path.join(FIG_DIR, "sbr_moca_stratification.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ============================================================
# LOGISTIC REGRESSION + ROC (UNCHANGED)
# ============================================================

if len(merged) > 10:

    X = merged[["MEAN_STRIATAL_SBR", "NP3TOT"]]
    y = merged["COGNITIVE_IMPAIRMENT"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )

    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    print("\n=== LOGISTIC REGRESSION RESULTS ===")
    print(classification_report(y_test, preds, zero_division=0))

    fpr, tpr, _ = roc_curve(y_test, probs)
    roc_auc = auc(fpr, tpr)

    plt.figure()

    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], "--")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        os.path.join(FIG_DIR, "roc_curve.png"),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"\nAUC: {roc_auc:.3f}")

print("\nPipeline complete.")