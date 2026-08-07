"""
HECKTOR 2026 - Comprehensive Results Summary
Generates tables and figures for 5-fold CV + ensemble test results.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import json
from lifelines.utils import concordance_index
from sklearn.metrics import balanced_accuracy_score

# ============================================================
# PATHS
# ============================================================
BASE = r"Z:\HOMES\HECKTOR_2026\task2_3_prediction"
PRED_DIR = os.path.join(BASE, "results", "predictions")
CKPT_DIR = os.path.join(BASE, "results", "checkpoints")
OUT_DIR  = os.path.join(BASE, "results", "summary")
os.makedirs(OUT_DIR, exist_ok=True)

PYTHON = r"C:\Users\bma2\AppData\Local\miniconda3\envs\xai_cuda118\python.exe"

# ============================================================
# MODEL NAMES (ordered for display, short labels)
# ============================================================
MODELS = [
    "clinical",
    "CT+PET",
    "clinical+CT+PET",
    "prob_T+prob_N",
    "CT+prob_T+prob_N",
    "PET+prob_T+prob_N",
    "CT+PET+prob_T+prob_N",
    "clinical+prob_T+prob_N",
    "clinical+CT+prob_T+prob_N",
    "CT+PET+hard_T+hard_N",
    "clinical+CT+PET+hard_T+hard_N",
    "clinical+CT+PET+prob_T+prob_N",
    "CT+PET+prob_T+prob_N+hard_T+hard_N",
    "clinical+CT+PET+prob_T+prob_N+hard_T+hard_N",
]

SHORT = {
    "clinical":                                    "Clin",
    "CT+PET":                                      "CT+PET",
    "clinical+CT+PET":                             "Clin+CT+PET",
    "prob_T+prob_N":                               "pTN",
    "CT+prob_T+prob_N":                            "CT+pTN",
    "PET+prob_T+prob_N":                           "PET+pTN",
    "CT+PET+prob_T+prob_N":                        "CT+PET+pTN",
    "clinical+prob_T+prob_N":                      "Clin+pTN",
    "clinical+CT+prob_T+prob_N":                   "Clin+CT+pTN",
    "CT+PET+hard_T+hard_N":                        "CT+PET+hTN",
    "clinical+CT+PET+hard_T+hard_N":               "Clin+CT+PET+hTN",
    "clinical+CT+PET+prob_T+prob_N":               "Clin+CT+PET+pTN",
    "CT+PET+prob_T+prob_N+hard_T+hard_N":          "CT+PET+pTN+hTN",
    "clinical+CT+PET+prob_T+prob_N+hard_T+hard_N": "Clin+CT+PET+pTN+hTN",
}

# ============================================================
# 5-FOLD CV DATA (from summary.json files)
# ============================================================
TASK2_FOLDS = {
    "clinical":                                    [0.30924904344021986, 0.3244773455710956, 0.33087254807456046, 0.31576607346911345, 0.3110790739550662],
    "CT+PET":                                      [0.5401549001394202, 0.5602850057534722, 0.5404727883056056, 0.5306335492706863, 0.5472736363621014],
    "clinical+CT+PET":                             [0.5448239722523778, 0.5266077200174296, 0.5382025718519526, 0.49035842869866236, 0.5186524765691075],
    "prob_T+prob_N":                               [0.5673940900430838, 0.5729182365766416, 0.5923826109391124, 0.5617994324292532, 0.5543869321030004],
    "CT+prob_T+prob_N":                            [0.5928812308262772, 0.5855783938346892, 0.5800585551359545, 0.5713014636687503, 0.5637044125245876],
    "PET+prob_T+prob_N":                           [0.5786135411715907, 0.5596280727930569, 0.6117801574070925, 0.5812329959837332, 0.5704354040833872],
    "CT+PET+prob_T+prob_N":                        [0.608434986066565, 0.5675717544705748, 0.6174563246351172, 0.5574449758855953, 0.5744546808352041],
    "clinical+prob_T+prob_N":                      [0.5738691026036227, 0.5710326920697271, 0.5838528967592436, 0.5303561285877529, 0.5358339902390178],
    "clinical+CT+prob_T+prob_N":                   [0.5786828120709699, 0.5753340570122558, 0.5912875372821194, 0.5412189537541171, 0.534272795154946],
    "CT+PET+hard_T+hard_N":                        [0.605030310032245, 0.6090654137529137, 0.6143449923829181, 0.565582379680043, 0.5788079776182268],
    "clinical+CT+PET+hard_T+hard_N":               [0.5958845742320974, 0.5522860004484325, 0.5892158044054329, 0.5457038759494521, 0.5440293748863192],
    "clinical+CT+PET+prob_T+prob_N":               [0.5949220080315282, 0.5917491811349571, 0.5840945217493205, 0.5328371920582107, 0.5493095690578094],
    "CT+PET+prob_T+prob_N+hard_T+hard_N":          [0.6017355558957727, 0.5973177774093723, 0.6053137888083708, 0.5499635553202931, 0.5627191331522643],
    "clinical+CT+PET+prob_T+prob_N+hard_T+hard_N": [0.5884032945468705, 0.574214773996421, 0.6008751847567637, 0.539816159171876, 0.5616833419996035],
}

TASK3_FOLDS = {
    "clinical":                                    [0.6034545454545455, 0.6146643618553731, 0.6822973464194838, 0.6318443804034583, 0.638733705772812],
    "CT+PET":                                      [0.672, 0.6871218668971478, 0.72264631043257, 0.6963256484149856, 0.6787709497206704],
    "clinical+CT+PET":                             [0.6894545454545454, 0.6614808412561222, 0.7339149400218102, 0.640850144092219, 0.686530105524519],
    "prob_T+prob_N":                               [0.6476363636363637, 0.7323537885335638, 0.7295528898582334, 0.6660662824207493, 0.7101179391682185],
    "CT+prob_T+prob_N":                            [0.6254545454545455, 0.7136271967732641, 0.7037440930570702, 0.6469740634005764, 0.6682184978274364],
    "PET+prob_T+prob_N":                           [0.672, 0.6986459233650245, 0.7829880043620502, 0.6761527377521613, 0.6412166356300435],
    "CT+PET+prob_T+prob_N":                        [0.6541818181818182, 0.7450302506482281, 0.7684478371501272, 0.6981268011527377, 0.6334574798261949],
    "clinical+prob_T+prob_N":                      [0.6381818181818182, 0.6992221261884183, 0.7386404943656852, 0.6347262247838616, 0.6977032898820609],
    "clinical+CT+prob_T+prob_N":                   [0.6523636363636364, 0.7202535292422932, 0.7146492184660124, 0.6264409221902018, 0.6728739913097455],
    "CT+PET+hard_T+hard_N":                        [0.7032727272727273, 0.7248631518294439, 0.7739003998545984, 0.6545389048991355, 0.6992551210428305],
    "clinical+CT+PET+hard_T+hard_N":               [0.6527272727272727, 0.7216940363007779, 0.7030170846964741, 0.6930835734870316, 0.6750465549348231],
    "clinical+CT+PET+prob_T+prob_N":               [0.6345454545454545, 0.7116104868913857, 0.7320974191203199, 0.6786743515850144, 0.6554934823091247],
    "CT+PET+prob_T+prob_N+hard_T+hard_N":          [0.653090909090909, 0.7015269374819937, 0.7699018538713195, 0.6703890489913544, 0.6204220980757293],
    "clinical+CT+PET+prob_T+prob_N+hard_T+hard_N": [0.6432727272727272, 0.7150677038317488, 0.7288258814976373, 0.6840778097982709, 0.6641837368094351],
}

MT_BACC_FOLDS = {
    "clinical":                                    [0.281936640793945, 0.2838482074752098, 0.3169146825396825, 0.27626929109267206, 0.26300959072846897],
    "CT+PET":                                      [0.42913313510452067, 0.4904337921499776, 0.4978959176263097, 0.5162279572629657, 0.49486103461821657],
    "clinical+CT+PET":                             [0.5374955269797137, 0.5049834416673242, 0.5129075241207595, 0.4881793954806991, 0.5041185871811728],
    "prob_T+prob_N":                               [0.5185836615693544, 0.5412483396459664, 0.6133917224297126, 0.5301662002802667, 0.5254452379965375],
    "CT+prob_T+prob_N":                            [0.5529770041705282, 0.5477966033430653, 0.5460471457101359, 0.5373805753983425, 0.5524700542655399],
    "PET+prob_T+prob_N":                           [0.5620456183709196, 0.5802494681935348, 0.5841211484593838, 0.5550396380400585, 0.5871158452810299],
    "CT+PET+prob_T+prob_N":                        [0.5780939623880137, 0.5900750323851232, 0.5450476255316941, 0.5498807662916158, 0.5697787307311931],
    "clinical+prob_T+prob_N":                      [0.5197205649521162, 0.553685768135669, 0.5860619942421413, 0.5257685691874086, 0.5765943093234475],
    "clinical+CT+prob_T+prob_N":                   [0.5396028344145813, 0.5626134136878929, 0.6019251219006121, 0.5166327458169342, 0.5568757365525491],
    "CT+PET+hard_T+hard_N":                        [0.5614410687622283, 0.5829008850837738, 0.5644709967320263, 0.49990996411307603, 0.5639663194382757],
    "clinical+CT+PET+hard_T+hard_N":               [0.5423993409535579, 0.5178849185604566, 0.5828846353356156, 0.5068802297171726, 0.5319744617367737],
    "clinical+CT+PET+prob_T+prob_N":               [0.566694421918443, 0.555788355822089, 0.5498873391949373, 0.5396542975916399, 0.513928122341255],
    "CT+PET+prob_T+prob_N+hard_T+hard_N":          [0.5668138097260838, 0.5617302080867461, 0.6111514744786803, 0.52179334319, 0.582990842566766],
    "clinical+CT+PET+prob_T+prob_N+hard_T+hard_N": [0.5729406246782, 0.5581837822864884, 0.5765323490507314, 0.5607379160259732, 0.5369535939816378],
}

MT_CI_FOLDS = {
    "clinical":                                    [0.6068540623796689, 0.6329387990762124, 0.6713922210105416, 0.6212178517397882, 0.6232068855594517],
    "CT+PET":                                      [0.6542164035425491, 0.7188221709006929, 0.7099236641221374, 0.6077912254160364, 0.6436085431941345],
    "clinical+CT+PET":                             [0.6087793608009241, 0.6991916859122402, 0.7117411850236278, 0.6214069591527988, 0.5941982786101371],
    "prob_T+prob_N":                               [0.6900269541778976, 0.7312355658198614, 0.6895674300254453, 0.6172465960665658, 0.6241632132610775],
    "CT+prob_T+prob_N":                            [0.620331151328456, 0.75, 0.6753907669938204, 0.5786686838124054, 0.6248007650621613],
    "PET+prob_T+prob_N":                           [0.6788602233346168, 0.7058314087759815, 0.712831697564522, 0.575642965204236, 0.6069493146318138],
    "CT+PET+prob_T+prob_N":                        [0.6584520600693108, 0.6659930715935335, 0.7288258814976373, 0.602874432677761, 0.6477526299011794],
    "clinical+prob_T+prob_N":                      [0.6087793608009241, 0.7182448036951501, 0.708106143220647, 0.5779122541603631, 0.6678355116353204],
    "clinical+CT+prob_T+prob_N":                   [0.6288024643819792, 0.6830254041570438, 0.6895674300254453, 0.5983358547655068, 0.6225693337583679],
    "CT+PET+hard_T+hard_N":                        [0.6434347323835194, 0.6795612009237876, 0.7011995637949836, 0.6319969742813918, 0.6952502390819254],
    "clinical+CT+PET+hard_T+hard_N":               [0.6415094339622641, 0.7165127020785219, 0.6855688840421665, 0.6263237518910741, 0.6955690149824674],
    "clinical+CT+PET+prob_T+prob_N":               [0.6311128224874856, 0.6798498845265589, 0.7291893856779353, 0.5726172465960666, 0.6990755498884285],
    "CT+PET+prob_T+prob_N+hard_T+hard_N":          [0.6445899114362726, 0.6928406466512702, 0.7302798982188295, 0.6183812405446294, 0.6675167357347784],
    "clinical+CT+PET+prob_T+prob_N+hard_T+hard_N": [0.6649980747015788, 0.6939953810623557, 0.6870229007633588, 0.5718608169440242, 0.6990755498884285],
}

# ============================================================
# LOAD TEST LABELS
# ============================================================
test_label_path = os.path.join(BASE, "Task 2 radiomics model (RF)", "features", "test_dataset.csv")
test_meta = pd.read_csv(test_label_path)[["PatientID", "T_label", "N_label", "Relapse", "RFS"]]
# T_label uses 1-indexed (T1=1..T4=4), convert to 0-indexed to match T_pred
test_meta["T_label_0idx"] = test_meta["T_label"].astype(int) - 1
test_meta["N_label_0idx"] = test_meta["N_label"].astype(int)
print(f"Test set: {len(test_meta)} patients")
print(test_meta[["PatientID","T_label","T_label_0idx","N_label","N_label_0idx","Relapse","RFS"]].head(3))

# Verify label encoding by cross-checking with OOF data for a known model
train_data = pd.read_csv(os.path.join(BASE, "HECKTOR_2026_training_data.csv"))
oof_sample = pd.read_csv(os.path.join(PRED_DIR, "task2", "CT+PET", "oof_predictions.csv"))
oof_sample = oof_sample.merge(train_data[["PatientID","T-stage","N-stage"]], on="PatientID")
# T_pred=0 should be T1, T_pred=1 T2, etc.
print("\nLabel encoding check (T_pred vs T-stage):")
print(oof_sample[["PatientID","T_pred","T-stage","N_pred","N-stage"]].head(5))

# ============================================================
# COMPUTE ENSEMBLE TEST METRICS
# ============================================================

def ensemble_test_task2(model_name, task="task2"):
    pred_path = os.path.join(PRED_DIR, task, model_name, "test_predictions.csv")
    if not os.path.exists(pred_path):
        return None, None
    df = pd.read_csv(pred_path)
    df = df.merge(test_meta, on="PatientID")
    # Average probabilities (already ensemble from all 5 folds via single test_predictions.csv)
    t_pred = df[["T_prob_T1","T_prob_T2","T_prob_T3","T_prob_T4"]].values.argmax(axis=1)
    n_pred = df[["N_prob_N0","N_prob_N1","N_prob_N2","N_prob_N3"]].values.argmax(axis=1)
    t_bacc = balanced_accuracy_score(df["T_label_0idx"], t_pred)
    n_bacc = balanced_accuracy_score(df["N_label_0idx"], n_pred)
    return t_bacc, n_bacc

def ensemble_test_task3(model_name, task="task3"):
    pred_path = os.path.join(PRED_DIR, task, model_name, "test_predictions.csv")
    if not os.path.exists(pred_path):
        return None
    df = pd.read_csv(pred_path)
    df = df.merge(test_meta, on="PatientID")
    # Remove patients with NaN survival
    df = df.dropna(subset=["RFS","Relapse"])
    ci = concordance_index(df["RFS"], -df["risk_score"], df["Relapse"])
    return ci

# Compute test metrics
print("\nComputing ensemble test metrics...")
test_t2 = {}
test_t3 = {}
test_mt_t2 = {}
test_mt_t3 = {}

for m in MODELS:
    t_b, n_b = ensemble_test_task2(m, "task2")
    if t_b is not None:
        test_t2[m] = {"t_bacc": t_b, "n_bacc": n_b, "mean_bacc": (t_b+n_b)/2}
    ci = ensemble_test_task3(m, "task3")
    if ci is not None:
        test_t3[m] = ci
    # Multitask
    t_b_mt, n_b_mt = ensemble_test_task2(m, "multitask")
    if t_b_mt is not None:
        test_mt_t2[m] = {"t_bacc": t_b_mt, "n_bacc": n_b_mt, "mean_bacc": (t_b_mt+n_b_mt)/2}
    ci_mt = ensemble_test_task3(m, "multitask")
    if ci_mt is not None:
        test_mt_t3[m] = ci_mt

# Cox baselines for test
cox_test_ci = 0.5262948207171314  # from summary.json mean_test_ci
cox_reduced_test_ci = 0.5557768924302788

# ============================================================
# BUILD SUMMARY TABLES
# ============================================================

rows_t2 = []
for m in MODELS:
    f = np.array(TASK2_FOLDS[m])
    row = {"Model": m, "Short": SHORT[m]}
    for i, v in enumerate(f):
        row[f"F{i+1}"] = v
    row["CV Mean"] = f.mean()
    row["CV Std"] = f.std()
    row["CV Mean±Std"] = f"{f.mean():.4f}±{f.std():.4f}"
    if m in test_t2:
        row["Test T-BACC"] = test_t2[m]["t_bacc"]
        row["Test N-BACC"] = test_t2[m]["n_bacc"]
        row["Test Mean BACC"] = test_t2[m]["mean_bacc"]
    rows_t2.append(row)
df_t2 = pd.DataFrame(rows_t2)

rows_t3 = []
for m in MODELS:
    f = np.array(TASK3_FOLDS[m])
    row = {"Model": m, "Short": SHORT[m]}
    for i, v in enumerate(f):
        row[f"F{i+1}"] = v
    row["CV Mean"] = f.mean()
    row["CV Std"] = f.std()
    row["CV Mean±Std"] = f"{f.mean():.4f}±{f.std():.4f}"
    if m in test_t3:
        row["Test C-Index"] = test_t3[m]
    rows_t3.append(row)
df_t3 = pd.DataFrame(rows_t3)

rows_mt = []
for m in MODELS:
    fb = np.array(MT_BACC_FOLDS[m])
    fc = np.array(MT_CI_FOLDS[m])
    row = {"Model": m, "Short": SHORT[m]}
    row["CV BACC"] = fb.mean()
    row["CV BACC Std"] = fb.std()
    row["CV BACC Mean±Std"] = f"{fb.mean():.4f}±{fb.std():.4f}"
    row["CV C-Index"] = fc.mean()
    row["CV C-Index Std"] = fc.std()
    row["CV C-Index Mean±Std"] = f"{fc.mean():.4f}±{fc.std():.4f}"
    if m in test_mt_t2:
        row["Test BACC"] = test_mt_t2[m]["mean_bacc"]
    if m in test_mt_t3:
        row["Test C-Index"] = test_mt_t3[m]
    rows_mt.append(row)
df_mt = pd.DataFrame(rows_mt)

# Comparison table
rows_cmp = []
for m in MODELS:
    fb2 = np.array(TASK2_FOLDS[m])
    fb3 = np.array(TASK3_FOLDS[m])
    fmb = np.array(MT_BACC_FOLDS[m])
    fmc = np.array(MT_CI_FOLDS[m])
    row = {
        "Model": SHORT[m],
        "T2 CV BACC": f"{fb2.mean():.4f}±{fb2.std():.4f}",
        "MT CV BACC": f"{fmb.mean():.4f}±{fmb.std():.4f}",
        "ΔBACC": fmb.mean() - fb2.mean(),
        "T3 CV C-Index": f"{fb3.mean():.4f}±{fb3.std():.4f}",
        "MT CV C-Index": f"{fmc.mean():.4f}±{fmc.std():.4f}",
        "ΔC-Index": fmc.mean() - fb3.mean(),
    }
    if m in test_t2 and m in test_mt_t2:
        row["T2 Test BACC"] = f"{test_t2[m]['mean_bacc']:.4f}"
        row["MT Test BACC"] = f"{test_mt_t2[m]['mean_bacc']:.4f}"
    if m in test_t3 and m in test_mt_t3:
        row["T3 Test C-Index"] = f"{test_t3[m]:.4f}"
        row["MT Test C-Index"] = f"{test_mt_t3[m]:.4f}"
    rows_cmp.append(row)
df_cmp = pd.DataFrame(rows_cmp)

# Save tables
df_t2.to_csv(os.path.join(OUT_DIR, "table1_task2_results.csv"), index=False)
df_t3.to_csv(os.path.join(OUT_DIR, "table2_task3_results.csv"), index=False)
df_mt.to_csv(os.path.join(OUT_DIR, "table3_multitask_results.csv"), index=False)
df_cmp.to_csv(os.path.join(OUT_DIR, "table4_comparison.csv"), index=False)
print("Tables saved.")

# ============================================================
# PRINT SUMMARY TO CONSOLE
# ============================================================
print("\n" + "="*80)
print("TABLE 1: TASK 2 - T/N Stage Classification (5-fold CV BACC + Ensemble Test)")
print("="*80)
disp_t2 = df_t2[["Short","CV Mean±Std","Test T-BACC","Test N-BACC","Test Mean BACC"]].copy()
disp_t2 = disp_t2.sort_values("CV Mean±Std", key=lambda x: x.str.split("±").str[0].astype(float), ascending=False)
print(disp_t2.to_string(index=False))

print("\n" + "="*80)
print("TABLE 2: TASK 3 - Overall Survival (5-fold CV C-Index + Ensemble Test)")
print("="*80)
disp_t3 = df_t3[["Short","CV Mean±Std","Test C-Index"]].copy()
disp_t3 = disp_t3.sort_values("CV Mean±Std", key=lambda x: x.str.split("±").str[0].astype(float), ascending=False)
disp_t3.loc[len(disp_t3)] = {"Short":"Cox Full (baseline)", "CV Mean±Std":"0.6023±0.0293", "Test C-Index":cox_test_ci}
disp_t3.loc[len(disp_t3)] = {"Short":"Cox Reduced (baseline)", "CV Mean±Std":"0.5957±0.0258", "Test C-Index":cox_reduced_test_ci}
print(disp_t3.to_string(index=False))

print("\n" + "="*80)
print("TABLE 3: MULTI-TASK - Joint Task 2+3 (5-fold CV BACC & C-Index + Ensemble Test)")
print("="*80)
disp_mt = df_mt[["Short","CV BACC Mean±Std","CV C-Index Mean±Std","Test BACC","Test C-Index"]].copy()
print(disp_mt.to_string(index=False))

print("\n" + "="*80)
print("TABLE 4: COMPARISON - Single-task vs Multi-task (5-fold CV)")
print("="*80)
disp_cmp = df_cmp[["Model","T2 CV BACC","MT CV BACC","ΔBACC","T3 CV C-Index","MT CV C-Index","ΔC-Index"]].copy()
disp_cmp["ΔBACC"] = disp_cmp["ΔBACC"].map(lambda x: f"{x:+.4f}" if isinstance(x,float) else x)
disp_cmp["ΔC-Index"] = disp_cmp["ΔC-Index"].map(lambda x: f"{x:+.4f}" if isinstance(x,float) else x)
print(disp_cmp.to_string(index=False))

# ============================================================
# FIGURE 1: Task 2 BACC - Single-task vs Multi-task (CV) + Test
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(20, 7), sharey=False)

# Sort models by task2 CV BACC for display
t2_sorted = sorted(MODELS, key=lambda m: np.mean(TASK2_FOLDS[m]), reverse=True)
short_labels = [SHORT[m] for m in t2_sorted]
t2_cv_means  = [np.mean(TASK2_FOLDS[m]) for m in t2_sorted]
t2_cv_stds   = [np.std(TASK2_FOLDS[m]) for m in t2_sorted]
mt_cv_means  = [np.mean(MT_BACC_FOLDS[m]) for m in t2_sorted]
mt_cv_stds   = [np.std(MT_BACC_FOLDS[m]) for m in t2_sorted]

x = np.arange(len(t2_sorted))
w = 0.35

# Panel A: 5-fold CV BACC
ax = axes[0]
b1 = ax.bar(x - w/2, t2_cv_means, w, yerr=t2_cv_stds, capsize=3,
            label="Task 2 (single-task)", color="#2196F3", alpha=0.85, error_kw={"linewidth":1})
b2 = ax.bar(x + w/2, mt_cv_means, w, yerr=mt_cv_stds, capsize=3,
            label="Task 2 (multi-task)", color="#FF5722", alpha=0.85, error_kw={"linewidth":1})
ax.axhline(0.3300, color="black", linestyle="--", linewidth=1.2, label="Clinical LogReg baseline (0.3300)")
# Highlight best model
_all_a1 = list(t2_cv_means) + list(mt_cv_means)
_best_a1 = int(np.argmax(_all_a1))
if _best_a1 < len(t2_cv_means):
    b1[_best_a1].set_edgecolor("gold"); b1[_best_a1].set_linewidth(2.5)
    ax.text(x[_best_a1] - w/2, t2_cv_means[_best_a1] + t2_cv_stds[_best_a1] + 0.008,
            f"{t2_cv_means[_best_a1]:.4f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
else:
    _bi = _best_a1 - len(t2_cv_means)
    b2[_bi].set_edgecolor("gold"); b2[_bi].set_linewidth(2.5)
    ax.text(x[_bi] + w/2, mt_cv_means[_bi] + mt_cv_stds[_bi] + 0.008,
            f"{mt_cv_means[_bi]:.4f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(short_labels, rotation=45, ha="right", fontsize=8.5)
ax.set_ylabel("Mean Balanced Accuracy (BACC)", fontsize=11)
ax.set_title("(A) 5-fold CV: BACC", fontsize=12, fontweight="bold")
ax.set_ylim(0.18, 0.72)
ax.legend(fontsize=9, loc="lower right")
ax.grid(axis="y", alpha=0.3)
ax.set_xlabel("Model Configuration", fontsize=10)

# Panel B: Ensemble Test BACC
ax2 = axes[1]
models_with_test = [m for m in t2_sorted if m in test_t2 and m in test_mt_t2]
short_test = [SHORT[m] for m in models_with_test]
t2_test_vals = [test_t2[m]["mean_bacc"] for m in models_with_test]
mt_test_vals = [test_mt_t2[m]["mean_bacc"] for m in models_with_test]
x2 = np.arange(len(models_with_test))
b3 = ax2.bar(x2 - w/2, t2_test_vals, w, label="Task 2 (single-task)", color="#2196F3", alpha=0.85)
b4 = ax2.bar(x2 + w/2, mt_test_vals, w, label="Task 2 (multi-task)", color="#FF5722", alpha=0.85)
# Highlight best model
_all_b1 = list(t2_test_vals) + list(mt_test_vals)
_best_b1 = int(np.argmax(_all_b1))
if _best_b1 < len(t2_test_vals):
    b3[_best_b1].set_edgecolor("gold"); b3[_best_b1].set_linewidth(2.5)
    ax2.text(x2[_best_b1] - w/2, t2_test_vals[_best_b1] + 0.005,
             f"{t2_test_vals[_best_b1]:.4f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
else:
    _bi = _best_b1 - len(t2_test_vals)
    b4[_bi].set_edgecolor("gold"); b4[_bi].set_linewidth(2.5)
    ax2.text(x2[_bi] + w/2, mt_test_vals[_bi] + 0.005,
             f"{mt_test_vals[_bi]:.4f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
ax2.set_xticks(x2)
ax2.set_xticklabels(short_test, rotation=45, ha="right", fontsize=8.5)
ax2.set_ylabel("Ensemble Test BACC", fontsize=11)
ax2.set_title("(B) Ensemble Test: BACC", fontsize=12, fontweight="bold")
ax2.legend(fontsize=9, loc="lower right")
ax2.grid(axis="y", alpha=0.3)
ax2.set_xlabel("Model Configuration", fontsize=10)

fig.suptitle("Figure 1: Task 2 — T/N Stage Classification\nSingle-task vs Multi-task Learning", fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "figure1_task2_bacc.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("\nFigure 1 saved.")

# ============================================================
# FIGURE 2: Task 3 C-Index - Single-task vs Multi-task (CV) + Test
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(20, 7), sharey=False)

t3_sorted = sorted(MODELS, key=lambda m: np.mean(TASK3_FOLDS[m]), reverse=True)
short_labels3 = [SHORT[m] for m in t3_sorted]
t3_cv_means = [np.mean(TASK3_FOLDS[m]) for m in t3_sorted]
t3_cv_stds  = [np.std(TASK3_FOLDS[m]) for m in t3_sorted]
mt_ci_means = [np.mean(MT_CI_FOLDS[m]) for m in t3_sorted]
mt_ci_stds  = [np.std(MT_CI_FOLDS[m]) for m in t3_sorted]

x3 = np.arange(len(t3_sorted))

# Panel A: 5-fold CV C-Index
ax = axes[0]
c1 = ax.bar(x3 - w/2, t3_cv_means, w, yerr=t3_cv_stds, capsize=3,
            label="Task 3 (single-task)", color="#4CAF50", alpha=0.85, error_kw={"linewidth":1})
c2 = ax.bar(x3 + w/2, mt_ci_means, w, yerr=mt_ci_stds, capsize=3,
            label="Task 3 (multi-task)", color="#9C27B0", alpha=0.85, error_kw={"linewidth":1})
ax.axhline(0.6023, color="black", linestyle="--", linewidth=1.2, label="Cox Full baseline (0.6023)")
ax.axhline(0.5, color="gray", linestyle=":", linewidth=1.0, alpha=0.7, label="Random (0.5)")
# Highlight best model
_all_a2 = list(t3_cv_means) + list(mt_ci_means)
_best_a2 = int(np.argmax(_all_a2))
if _best_a2 < len(t3_cv_means):
    c1[_best_a2].set_edgecolor("gold"); c1[_best_a2].set_linewidth(2.5)
    ax.text(x3[_best_a2] - w/2, t3_cv_means[_best_a2] + t3_cv_stds[_best_a2] + 0.008,
            f"{t3_cv_means[_best_a2]:.4f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
else:
    _bi = _best_a2 - len(t3_cv_means)
    c2[_bi].set_edgecolor("gold"); c2[_bi].set_linewidth(2.5)
    ax.text(x3[_bi] + w/2, mt_ci_means[_bi] + mt_ci_stds[_bi] + 0.008,
            f"{mt_ci_means[_bi]:.4f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
ax.set_xticks(x3)
ax.set_xticklabels(short_labels3, rotation=45, ha="right", fontsize=8.5)
ax.set_ylabel("Mean Concordance Index (C-Index)", fontsize=11)
ax.set_title("(A) 5-fold CV: C-Index", fontsize=12, fontweight="bold")
ax.set_ylim(0.4, 0.83)
ax.legend(fontsize=9, loc="lower right")
ax.grid(axis="y", alpha=0.3)
ax.set_xlabel("Model Configuration", fontsize=10)

# Panel B: Ensemble Test C-Index
ax2 = axes[1]
models_with_test3 = [m for m in t3_sorted if m in test_t3 and m in test_mt_t3]
short_test3 = [SHORT[m] for m in models_with_test3]
t3_test_vals  = [test_t3[m] for m in models_with_test3]
mt_test_vals3 = [test_mt_t3[m] for m in models_with_test3]
x4 = np.arange(len(models_with_test3))
c3 = ax2.bar(x4 - w/2, t3_test_vals, w, label="Task 3 (single-task)", color="#4CAF50", alpha=0.85)
c4 = ax2.bar(x4 + w/2, mt_test_vals3, w, label="Task 3 (multi-task)", color="#9C27B0", alpha=0.85)
# Highlight best model
_all_b2 = list(t3_test_vals) + list(mt_test_vals3)
_best_b2 = int(np.argmax(_all_b2))
if _best_b2 < len(t3_test_vals):
    c3[_best_b2].set_edgecolor("gold"); c3[_best_b2].set_linewidth(2.5)
    ax2.text(x4[_best_b2] - w/2, t3_test_vals[_best_b2] + 0.005,
             f"{t3_test_vals[_best_b2]:.4f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
else:
    _bi = _best_b2 - len(t3_test_vals)
    c4[_bi].set_edgecolor("gold"); c4[_bi].set_linewidth(2.5)
    ax2.text(x4[_bi] + w/2, mt_test_vals3[_bi] + 0.005,
             f"{mt_test_vals3[_bi]:.4f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
ax2.axhline(cox_test_ci, color="black", linestyle="--", linewidth=1.2, label=f"Cox Full test ({cox_test_ci:.4f})")
ax2.axhline(cox_reduced_test_ci, color="gray", linestyle="--", linewidth=1.2, label=f"Cox Reduced test ({cox_reduced_test_ci:.4f})")
ax2.set_xticks(x4)
ax2.set_xticklabels(short_test3, rotation=45, ha="right", fontsize=8.5)
ax2.set_ylabel("Ensemble Test C-Index", fontsize=11)
ax2.set_title("(B) Ensemble Test: C-Index", fontsize=12, fontweight="bold")
ax2.legend(fontsize=8.5, loc="lower right")
ax2.grid(axis="y", alpha=0.3)
ax2.set_xlabel("Model Configuration", fontsize=10)

fig.suptitle("Figure 2: Task 3 — Overall Survival Prediction\nSingle-task vs Multi-task Learning", fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "figure2_task3_cindex.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("Figure 2 saved.")

# ============================================================
# FIGURE 3: Multi-task BACC (CV + Test)
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(20, 7), sharey=False)

mt_sorted = sorted(MODELS, key=lambda m: np.mean(MT_BACC_FOLDS[m]), reverse=True)
short_mt   = [SHORT[m] for m in mt_sorted]
mt_b_means = [np.mean(MT_BACC_FOLDS[m]) for m in mt_sorted]
mt_b_stds  = [np.std(MT_BACC_FOLDS[m])  for m in mt_sorted]
xm = np.arange(len(mt_sorted))

# Panel A: 5-fold CV BACC
ax = axes[0]
d1 = ax.bar(xm, mt_b_means, w*1.5, yerr=mt_b_stds, capsize=3,
            color="#FF5722", alpha=0.85, error_kw={"linewidth": 1})
ax.axhline(0.3300, color="black", linestyle="--", linewidth=1.2, label="Clinical LogReg baseline (0.3300)")
_best_m3a = int(np.argmax(mt_b_means))
d1[_best_m3a].set_edgecolor("gold"); d1[_best_m3a].set_linewidth(2.5)
ax.text(xm[_best_m3a], mt_b_means[_best_m3a] + mt_b_stds[_best_m3a] + 0.008,
        f"{mt_b_means[_best_m3a]:.4f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
ax.set_xticks(xm)
ax.set_xticklabels(short_mt, rotation=45, ha="right", fontsize=8.5)
ax.set_ylabel("Mean Balanced Accuracy (BACC)", fontsize=11)
ax.set_title("(A) 5-fold CV: Multi-task BACC", fontsize=12, fontweight="bold")
ax.set_ylim(0.18, 0.72)
ax.legend(fontsize=9, loc="lower right")
ax.grid(axis="y", alpha=0.3)
ax.set_xlabel("Model Configuration", fontsize=10)

# Panel B: Ensemble Test BACC
ax2 = axes[1]
mt_test_models = [m for m in mt_sorted if m in test_mt_t2]
short_mt_test  = [SHORT[m] for m in mt_test_models]
mt_test_b_vals = [test_mt_t2[m]["mean_bacc"] for m in mt_test_models]
xmt = np.arange(len(mt_test_models))
d2 = ax2.bar(xmt, mt_test_b_vals, w*1.5, color="#FF5722", alpha=0.85)
_best_m3b = int(np.argmax(mt_test_b_vals))
d2[_best_m3b].set_edgecolor("gold"); d2[_best_m3b].set_linewidth(2.5)
ax2.text(xmt[_best_m3b], mt_test_b_vals[_best_m3b] + 0.005,
         f"{mt_test_b_vals[_best_m3b]:.4f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
ax2.set_xticks(xmt)
ax2.set_xticklabels(short_mt_test, rotation=45, ha="right", fontsize=8.5)
ax2.set_ylabel("Ensemble Test BACC", fontsize=11)
ax2.set_title("(B) Ensemble Test: Multi-task BACC", fontsize=12, fontweight="bold")
ax2.grid(axis="y", alpha=0.3)
ax2.set_xlabel("Model Configuration", fontsize=10)

fig.suptitle("Figure 3: Multi-task — T/N Stage Classification (BACC)", fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "figure3_multitask_bacc.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("\nFigure 3 saved.")

# ============================================================
# FIGURE 4: Multi-task C-Index (CV + Test)
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(20, 7), sharey=False)

mt_ci_sorted = sorted(MODELS, key=lambda m: np.mean(MT_CI_FOLDS[m]), reverse=True)
short_mt_ci   = [SHORT[m] for m in mt_ci_sorted]
mt_ci_cv_means = [np.mean(MT_CI_FOLDS[m]) for m in mt_ci_sorted]
mt_ci_cv_stds  = [np.std(MT_CI_FOLDS[m])  for m in mt_ci_sorted]
xmc = np.arange(len(mt_ci_sorted))

# Panel A: 5-fold CV C-Index
ax = axes[0]
e1 = ax.bar(xmc, mt_ci_cv_means, w*1.5, yerr=mt_ci_cv_stds, capsize=3,
            color="#9C27B0", alpha=0.85, error_kw={"linewidth": 1})
ax.axhline(0.6023, color="black", linestyle="--", linewidth=1.2, label="Cox Full baseline (0.6023)")
ax.axhline(0.5, color="gray", linestyle=":", linewidth=1.0, alpha=0.7, label="Random (0.5)")
_best_m4a = int(np.argmax(mt_ci_cv_means))
e1[_best_m4a].set_edgecolor("gold"); e1[_best_m4a].set_linewidth(2.5)
ax.text(xmc[_best_m4a], mt_ci_cv_means[_best_m4a] + mt_ci_cv_stds[_best_m4a] + 0.008,
        f"{mt_ci_cv_means[_best_m4a]:.4f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
ax.set_xticks(xmc)
ax.set_xticklabels(short_mt_ci, rotation=45, ha="right", fontsize=8.5)
ax.set_ylabel("Mean Concordance Index (C-Index)", fontsize=11)
ax.set_title("(A) 5-fold CV: Multi-task C-Index", fontsize=12, fontweight="bold")
ax.set_ylim(0.4, 0.83)
ax.legend(fontsize=9, loc="lower right")
ax.grid(axis="y", alpha=0.3)
ax.set_xlabel("Model Configuration", fontsize=10)

# Panel B: Ensemble Test C-Index
ax2 = axes[1]
mt_test_ci_models = [m for m in mt_ci_sorted if m in test_mt_t3]
short_mt_ci_test  = [SHORT[m] for m in mt_test_ci_models]
mt_test_ci_vals   = [test_mt_t3[m] for m in mt_test_ci_models]
xmct = np.arange(len(mt_test_ci_models))
e2 = ax2.bar(xmct, mt_test_ci_vals, w*1.5, color="#9C27B0", alpha=0.85)
ax2.axhline(cox_test_ci, color="black", linestyle="--", linewidth=1.2, label=f"Cox Full test ({cox_test_ci:.4f})")
ax2.axhline(cox_reduced_test_ci, color="gray", linestyle="--", linewidth=1.2, label=f"Cox Reduced test ({cox_reduced_test_ci:.4f})")
_best_m4b = int(np.argmax(mt_test_ci_vals))
e2[_best_m4b].set_edgecolor("gold"); e2[_best_m4b].set_linewidth(2.5)
ax2.text(xmct[_best_m4b], mt_test_ci_vals[_best_m4b] + 0.005,
         f"{mt_test_ci_vals[_best_m4b]:.4f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
ax2.set_xticks(xmct)
ax2.set_xticklabels(short_mt_ci_test, rotation=45, ha="right", fontsize=8.5)
ax2.set_ylabel("Ensemble Test C-Index", fontsize=11)
ax2.set_title("(B) Ensemble Test: Multi-task C-Index", fontsize=12, fontweight="bold")
ax2.legend(fontsize=8.5, loc="lower right")
ax2.grid(axis="y", alpha=0.3)
ax2.set_xlabel("Model Configuration", fontsize=10)

fig.suptitle("Figure 4: Multi-task — Overall Survival Prediction (C-Index)", fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "figure4_multitask_cindex.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("Figure 4 saved.")

print(f"\nAll outputs saved to: {OUT_DIR}")
