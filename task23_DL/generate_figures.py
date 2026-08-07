"""Generate figures from pre-computed CSV tables + 5-fold data."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from lifelines.utils import concordance_index
from sklearn.metrics import balanced_accuracy_score

BASE    = r"Z:\HOMES\HECKTOR_2026\task2_3_prediction"
PRED    = os.path.join(BASE, "results", "predictions")
OUT     = os.path.join(BASE, "results", "summary")
os.makedirs(OUT, exist_ok=True)

MODELS = [
    "clinical","CT+PET","clinical+CT+PET",
    "prob_T+prob_N","CT+prob_T+prob_N","PET+prob_T+prob_N",
    "CT+PET+prob_T+prob_N","clinical+prob_T+prob_N","clinical+CT+prob_T+prob_N",
    "CT+PET+hard_T+hard_N","clinical+CT+PET+hard_T+hard_N",
    "clinical+CT+PET+prob_T+prob_N","CT+PET+prob_T+prob_N+hard_T+hard_N",
    "clinical+CT+PET+prob_T+prob_N+hard_T+hard_N",
]
SHORT = {
    "clinical":"Clin","CT+PET":"CT+PET","clinical+CT+PET":"Clin+CT+PET",
    "prob_T+prob_N":"pTN","CT+prob_T+prob_N":"CT+pTN","PET+prob_T+prob_N":"PET+pTN",
    "CT+PET+prob_T+prob_N":"CT+PET+pTN","clinical+prob_T+prob_N":"Clin+pTN",
    "clinical+CT+prob_T+prob_N":"Clin+CT+pTN","CT+PET+hard_T+hard_N":"CT+PET+hTN",
    "clinical+CT+PET+hard_T+hard_N":"Clin+CT+PET+hTN",
    "clinical+CT+PET+prob_T+prob_N":"Clin+CT+PET+pTN",
    "CT+PET+prob_T+prob_N+hard_T+hard_N":"CT+PET+pTN+hTN",
    "clinical+CT+PET+prob_T+prob_N+hard_T+hard_N":"Clin+CT+PET+pTN+hTN",
}

TASK2_FOLDS = {
    "clinical":[0.30924904344021986,0.3244773455710956,0.33087254807456046,0.31576607346911345,0.3110790739550662],
    "CT+PET":[0.5401549001394202,0.5602850057534722,0.5404727883056056,0.5306335492706863,0.5472736363621014],
    "clinical+CT+PET":[0.5448239722523778,0.5266077200174296,0.5382025718519526,0.49035842869866236,0.5186524765691075],
    "prob_T+prob_N":[0.5673940900430838,0.5729182365766416,0.5923826109391124,0.5617994324292532,0.5543869321030004],
    "CT+prob_T+prob_N":[0.5928812308262772,0.5855783938346892,0.5800585551359545,0.5713014636687503,0.5637044125245876],
    "PET+prob_T+prob_N":[0.5786135411715907,0.5596280727930569,0.6117801574070925,0.5812329959837332,0.5704354040833872],
    "CT+PET+prob_T+prob_N":[0.608434986066565,0.5675717544705748,0.6174563246351172,0.5574449758855953,0.5744546808352041],
    "clinical+prob_T+prob_N":[0.5738691026036227,0.5710326920697271,0.5838528967592436,0.5303561285877529,0.5358339902390178],
    "clinical+CT+prob_T+prob_N":[0.5786828120709699,0.5753340570122558,0.5912875372821194,0.5412189537541171,0.534272795154946],
    "CT+PET+hard_T+hard_N":[0.605030310032245,0.6090654137529137,0.6143449923829181,0.565582379680043,0.5788079776182268],
    "clinical+CT+PET+hard_T+hard_N":[0.5958845742320974,0.5522860004484325,0.5892158044054329,0.5457038759494521,0.5440293748863192],
    "clinical+CT+PET+prob_T+prob_N":[0.5949220080315282,0.5917491811349571,0.5840945217493205,0.5328371920582107,0.5493095690578094],
    "CT+PET+prob_T+prob_N+hard_T+hard_N":[0.6017355558957727,0.5973177774093723,0.6053137888083708,0.5499635553202931,0.5627191331522643],
    "clinical+CT+PET+prob_T+prob_N+hard_T+hard_N":[0.5884032945468705,0.574214773996421,0.6008751847567637,0.539816159171876,0.5616833419996035],
}
TASK3_FOLDS = {
    "clinical":[0.6034545454545455,0.6146643618553731,0.6822973464194838,0.6318443804034583,0.638733705772812],
    "CT+PET":[0.672,0.6871218668971478,0.72264631043257,0.6963256484149856,0.6787709497206704],
    "clinical+CT+PET":[0.6894545454545454,0.6614808412561222,0.7339149400218102,0.640850144092219,0.686530105524519],
    "prob_T+prob_N":[0.6476363636363637,0.7323537885335638,0.7295528898582334,0.6660662824207493,0.7101179391682185],
    "CT+prob_T+prob_N":[0.6254545454545455,0.7136271967732641,0.7037440930570702,0.6469740634005764,0.6682184978274364],
    "PET+prob_T+prob_N":[0.672,0.6986459233650245,0.7829880043620502,0.6761527377521613,0.6412166356300435],
    "CT+PET+prob_T+prob_N":[0.6541818181818182,0.7450302506482281,0.7684478371501272,0.6981268011527377,0.6334574798261949],
    "clinical+prob_T+prob_N":[0.6381818181818182,0.6992221261884183,0.7386404943656852,0.6347262247838616,0.6977032898820609],
    "clinical+CT+prob_T+prob_N":[0.6523636363636364,0.7202535292422932,0.7146492184660124,0.6264409221902018,0.6728739913097455],
    "CT+PET+hard_T+hard_N":[0.7032727272727273,0.7248631518294439,0.7739003998545984,0.6545389048991355,0.6992551210428305],
    "clinical+CT+PET+hard_T+hard_N":[0.6527272727272727,0.7216940363007779,0.7030170846964741,0.6930835734870316,0.6750465549348231],
    "clinical+CT+PET+prob_T+prob_N":[0.6345454545454545,0.7116104868913857,0.7320974191203199,0.6786743515850144,0.6554934823091247],
    "CT+PET+prob_T+prob_N+hard_T+hard_N":[0.653090909090909,0.7015269374819937,0.7699018538713195,0.6703890489913544,0.6204220980757293],
    "clinical+CT+PET+prob_T+prob_N+hard_T+hard_N":[0.6432727272727272,0.7150677038317488,0.7288258814976373,0.6840778097982709,0.6641837368094351],
}
MT_BACC = {
    "clinical":[0.281936640793945,0.2838482074752098,0.3169146825396825,0.27626929109267206,0.26300959072846897],
    "CT+PET":[0.42913313510452067,0.4904337921499776,0.4978959176263097,0.5162279572629657,0.49486103461821657],
    "clinical+CT+PET":[0.5374955269797137,0.5049834416673242,0.5129075241207595,0.4881793954806991,0.5041185871811728],
    "prob_T+prob_N":[0.5185836615693544,0.5412483396459664,0.6133917224297126,0.5301662002802667,0.5254452379965375],
    "CT+prob_T+prob_N":[0.5529770041705282,0.5477966033430653,0.5460471457101359,0.5373805753983425,0.5524700542655399],
    "PET+prob_T+prob_N":[0.5620456183709196,0.5802494681935348,0.5841211484593838,0.5550396380400585,0.5871158452810299],
    "CT+PET+prob_T+prob_N":[0.5780939623880137,0.5900750323851232,0.5450476255316941,0.5498807662916158,0.5697787307311931],
    "clinical+prob_T+prob_N":[0.5197205649521162,0.553685768135669,0.5860619942421413,0.5257685691874086,0.5765943093234475],
    "clinical+CT+prob_T+prob_N":[0.5396028344145813,0.5626134136878929,0.6019251219006121,0.5166327458169342,0.5568757365525491],
    "CT+PET+hard_T+hard_N":[0.5614410687622283,0.5829008850837738,0.5644709967320263,0.49990996411307603,0.5639663194382757],
    "clinical+CT+PET+hard_T+hard_N":[0.5423993409535579,0.5178849185604566,0.5828846353356156,0.5068802297171726,0.5319744617367737],
    "clinical+CT+PET+prob_T+prob_N":[0.566694421918443,0.555788355822089,0.5498873391949373,0.5396542975916399,0.513928122341255],
    "CT+PET+prob_T+prob_N+hard_T+hard_N":[0.5668138097260838,0.5617302080867461,0.6111514744786803,0.52179334319,0.582990842566766],
    "clinical+CT+PET+prob_T+prob_N+hard_T+hard_N":[0.5729406246782,0.5581837822864884,0.5765323490507314,0.5607379160259732,0.5369535939816378],
}
MT_CI = {
    "clinical":[0.6068540623796689,0.6329387990762124,0.6713922210105416,0.6212178517397882,0.6232068855594517],
    "CT+PET":[0.6542164035425491,0.7188221709006929,0.7099236641221374,0.6077912254160364,0.6436085431941345],
    "clinical+CT+PET":[0.6087793608009241,0.6991916859122402,0.7117411850236278,0.6214069591527988,0.5941982786101371],
    "prob_T+prob_N":[0.6900269541778976,0.7312355658198614,0.6895674300254453,0.6172465960665658,0.6241632132610775],
    "CT+prob_T+prob_N":[0.620331151328456,0.75,0.6753907669938204,0.5786686838124054,0.6248007650621613],
    "PET+prob_T+prob_N":[0.6788602233346168,0.7058314087759815,0.712831697564522,0.575642965204236,0.6069493146318138],
    "CT+PET+prob_T+prob_N":[0.6584520600693108,0.6659930715935335,0.7288258814976373,0.602874432677761,0.6477526299011794],
    "clinical+prob_T+prob_N":[0.6087793608009241,0.7182448036951501,0.708106143220647,0.5779122541603631,0.6678355116353204],
    "clinical+CT+prob_T+prob_N":[0.6288024643819792,0.6830254041570438,0.6895674300254453,0.5983358547655068,0.6225693337583679],
    "CT+PET+hard_T+hard_N":[0.6434347323835194,0.6795612009237876,0.7011995637949836,0.6319969742813918,0.6952502390819254],
    "clinical+CT+PET+hard_T+hard_N":[0.6415094339622641,0.7165127020785219,0.6855688840421665,0.6263237518910741,0.6955690149824674],
    "clinical+CT+PET+prob_T+prob_N":[0.6311128224874856,0.6798498845265589,0.7291893856779353,0.5726172465960666,0.6990755498884285],
    "CT+PET+prob_T+prob_N+hard_T+hard_N":[0.6445899114362726,0.6928406466512702,0.7302798982188295,0.6183812405446294,0.6675167357347784],
    "clinical+CT+PET+prob_T+prob_N+hard_T+hard_N":[0.6649980747015788,0.6939953810623557,0.6870229007633588,0.5718608169440242,0.6990755498884285],
}

# Load test labels
test_meta = pd.read_csv(os.path.join(BASE,"Task 2 radiomics model (RF)","features","test_dataset.csv"))
test_meta = test_meta[["PatientID","T_label","N_label","Relapse","RFS"]].copy()
test_meta["T_label_0idx"] = test_meta["T_label"].astype(int) - 1
test_meta["N_label_0idx"] = test_meta["N_label"].astype(int)

def test_bacc(model, task="task2"):
    p = os.path.join(PRED, task, model, "test_predictions.csv")
    if not os.path.exists(p): return None
    df = pd.read_csv(p).merge(test_meta, on="PatientID")
    tp = df[["T_prob_T1","T_prob_T2","T_prob_T3","T_prob_T4"]].values.argmax(1)
    np_ = df[["N_prob_N0","N_prob_N1","N_prob_N2","N_prob_N3"]].values.argmax(1)
    tb = balanced_accuracy_score(df["T_label_0idx"], tp)
    nb = balanced_accuracy_score(df["N_label_0idx"], np_)
    return (tb+nb)/2

def test_ci(model, task="task3"):
    p = os.path.join(PRED, task, model, "test_predictions.csv")
    if not os.path.exists(p): return None
    df = pd.read_csv(p).merge(test_meta, on="PatientID").dropna(subset=["RFS","Relapse"])
    return concordance_index(df["RFS"], -df["risk_score"], df["Relapse"])

# Gather test data
tt2 = {m: test_bacc(m,"task2") for m in MODELS}
tt3 = {m: test_ci(m,"task3")   for m in MODELS}
mt2 = {m: test_bacc(m,"multitask") for m in MODELS}
mt3 = {m: test_ci(m,"multitask")   for m in MODELS}

COX_TEST_CI     = 0.5262948207171314
COX_RED_TEST_CI = 0.5557768924302788
LOGREG_CV       = 0.3300

# Sort by task2/task3 CV mean
t2_sorted = sorted(MODELS, key=lambda m: np.mean(TASK2_FOLDS[m]), reverse=True)
t3_sorted = sorted(MODELS, key=lambda m: np.mean(TASK3_FOLDS[m]), reverse=True)

w = 0.35

# ---------------------------------------------------------------
# FIGURE 1: Task 2 BACC
# ---------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(20, 7))
BLUE, ORANGE = "#2196F3", "#FF5722"

# Panel A - 5-fold CV
ax = axes[0]
x = np.arange(len(t2_sorted))
cv_m = [np.mean(TASK2_FOLDS[m]) for m in t2_sorted]
cv_s = [np.std(TASK2_FOLDS[m])  for m in t2_sorted]
mt_m = [np.mean(MT_BACC[m])     for m in t2_sorted]
mt_s = [np.std(MT_BACC[m])      for m in t2_sorted]
ax.bar(x-w/2, cv_m, w, yerr=cv_s, capsize=3, color=BLUE,   alpha=0.85, label="Task 2 only", error_kw={"linewidth":1.2})
ax.bar(x+w/2, mt_m, w, yerr=mt_s, capsize=3, color=ORANGE, alpha=0.85, label="Multi-task",  error_kw={"linewidth":1.2})
ax.axhline(LOGREG_CV, color="black", ls="--", lw=1.3, label=f"Clinical LogReg ({LOGREG_CV:.4f})")
ax.set_xticks(x); ax.set_xticklabels([SHORT[m] for m in t2_sorted], rotation=45, ha="right", fontsize=8.5)
ax.set_ylabel("Mean Balanced Accuracy (BACC)", fontsize=11)
ax.set_title("(A) 5-fold CV BACC", fontsize=12, fontweight="bold")
ax.set_ylim(0.15, 0.75); ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)
ax.set_xlabel("Model", fontsize=10)

# Panel B - Ensemble test
ax2 = axes[1]
ms = [m for m in t2_sorted if tt2.get(m) is not None and mt2.get(m) is not None]
x2 = np.arange(len(ms))
ax2.bar(x2-w/2, [tt2[m] for m in ms], w, color=BLUE,   alpha=0.85, label="Task 2 only")
ax2.bar(x2+w/2, [mt2[m] for m in ms], w, color=ORANGE, alpha=0.85, label="Multi-task")
ax2.set_xticks(x2); ax2.set_xticklabels([SHORT[m] for m in ms], rotation=45, ha="right", fontsize=8.5)
ax2.set_ylabel("Ensemble Test BACC", fontsize=11)
ax2.set_title("(B) Ensemble Test BACC", fontsize=12, fontweight="bold")
ax2.legend(fontsize=9); ax2.grid(axis="y", alpha=0.3)
ax2.set_xlabel("Model", fontsize=10)

fig.suptitle("Figure 1: Task 2 — T/N Stage Classification\nSingle-task vs Multi-task", fontsize=14, fontweight="bold")
plt.tight_layout()
out1 = os.path.join(OUT, "figure1_task2_bacc.png")
fig.savefig(out1, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved {out1}")

# ---------------------------------------------------------------
# FIGURE 2: Task 3 C-Index
# ---------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(20, 7))
GREEN, PURPLE = "#2E7D32", "#7B1FA2"

# Panel A - 5-fold CV
ax = axes[0]
x = np.arange(len(t3_sorted))
cv_m3 = [np.mean(TASK3_FOLDS[m]) for m in t3_sorted]
cv_s3 = [np.std(TASK3_FOLDS[m])  for m in t3_sorted]
mt_m3 = [np.mean(MT_CI[m])       for m in t3_sorted]
mt_s3 = [np.std(MT_CI[m])        for m in t3_sorted]
ax.bar(x-w/2, cv_m3, w, yerr=cv_s3, capsize=3, color=GREEN,  alpha=0.85, label="Task 3 only", error_kw={"linewidth":1.2})
ax.bar(x+w/2, mt_m3, w, yerr=mt_s3, capsize=3, color=PURPLE, alpha=0.85, label="Multi-task",  error_kw={"linewidth":1.2})
ax.axhline(0.6023, color="black", ls="--", lw=1.3, label="Cox Full CV (0.6023)")
ax.axhline(0.5, color="gray", ls=":", lw=1.0, alpha=0.7, label="Random (0.5)")
ax.set_xticks(x); ax.set_xticklabels([SHORT[m] for m in t3_sorted], rotation=45, ha="right", fontsize=8.5)
ax.set_ylabel("Mean Concordance Index (C-Index)", fontsize=11)
ax.set_title("(A) 5-fold CV C-Index", fontsize=12, fontweight="bold")
ax.set_ylim(0.40, 0.83); ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)
ax.set_xlabel("Model", fontsize=10)

# Panel B - Ensemble test
ax2 = axes[1]
ms3 = [m for m in t3_sorted if tt3.get(m) is not None and mt3.get(m) is not None]
x3 = np.arange(len(ms3))
ax2.bar(x3-w/2, [tt3[m] for m in ms3], w, color=GREEN,  alpha=0.85, label="Task 3 only")
ax2.bar(x3+w/2, [mt3[m] for m in ms3], w, color=PURPLE, alpha=0.85, label="Multi-task")
ax2.axhline(COX_TEST_CI,     color="black", ls="--", lw=1.3, label=f"Cox Full test ({COX_TEST_CI:.4f})")
ax2.axhline(COX_RED_TEST_CI, color="gray",  ls="--", lw=1.3, label=f"Cox Reduced test ({COX_RED_TEST_CI:.4f})")
ax2.set_xticks(x3); ax2.set_xticklabels([SHORT[m] for m in ms3], rotation=45, ha="right", fontsize=8.5)
ax2.set_ylabel("Ensemble Test C-Index", fontsize=11)
ax2.set_title("(B) Ensemble Test C-Index", fontsize=12, fontweight="bold")
ax2.legend(fontsize=8.5); ax2.grid(axis="y", alpha=0.3)
ax2.set_xlabel("Model", fontsize=10)

fig.suptitle("Figure 2: Task 3 — Overall Survival Prediction\nSingle-task vs Multi-task", fontsize=14, fontweight="bold")
plt.tight_layout()
out2 = os.path.join(OUT, "figure2_task3_cindex.png")
fig.savefig(out2, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved {out2}")

# ---------------------------------------------------------------
# Print comparison table (ASCII-safe)
# ---------------------------------------------------------------
print("\n=== TABLE 4: Single-task vs Multi-task (CV + Test) ===")
hdr = f"{'Model':<22} {'T2-CV':>14} {'MT-CV-BACC':>14} {'d-BACC':>8} {'T3-CV':>14} {'MT-CV-CI':>14} {'d-CI':>8} | {'T2-Test':>8} {'MT-Test-B':>10} {'T3-Test':>8} {'MT-Test-C':>10}"
print(hdr)
print("-"*len(hdr))
for m in MODELS:
    fb = np.array(TASK2_FOLDS[m]); fc = np.array(TASK3_FOLDS[m])
    mb = np.array(MT_BACC[m]);     mc = np.array(MT_CI[m])
    db = mb.mean()-fb.mean(); dc = mc.mean()-fc.mean()
    t2t  = f"{tt2[m]:.4f}" if tt2.get(m) else "N/A"
    mt2t = f"{mt2[m]:.4f}" if mt2.get(m) else "N/A"
    t3t  = f"{tt3[m]:.4f}" if tt3.get(m) else "N/A"
    mt3t = f"{mt3[m]:.4f}" if mt3.get(m) else "N/A"
    line = (f"{SHORT[m]:<22} {fb.mean():.4f}+/-{fb.std():.4f} {mb.mean():.4f}+/-{mb.std():.4f} "
            f"{db:+.4f} {fc.mean():.4f}+/-{fc.std():.4f} {mc.mean():.4f}+/-{mc.std():.4f} {dc:+.4f} | "
            f"{t2t:>8} {mt2t:>10} {t3t:>8} {mt3t:>10}")
    print(line)

print("\nDone.")
