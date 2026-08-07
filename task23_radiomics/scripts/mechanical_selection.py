import pandas as pd, numpy as np, glob, os
from sksurv.metrics import concordance_index_censored

base='/nvme/jin/HECKTOR26/task2_3_prediction/results/predictions'
lab=pd.read_csv('features/oof_clinical_rfs_v2.csv')[['PatientID','Relapse','RFS']]
rsf_df=pd.read_csv('features/oof_clinical_rfs_v2.csv')[['PatientID','rf_rfs_risk']].rename(columns={'rf_rfs_risk':'RSF'})

# ---- 收集所有DL配置的OOF risk ----
pool={}
for f in sorted(glob.glob(f'{base}/*/*/oof_predictions.csv')):
    d=pd.read_csv(f)
    if 'risk_score' not in d.columns: continue
    name=os.path.relpath(f,base).replace('/oof_predictions.csv','')
    pool[name]=d[['PatientID','risk_score']]

# 对齐所有源到同一批患者
m=lab.copy()
for name,d in pool.items():
    m=m.merge(d.rename(columns={'risk_score':name}),on='PatientID',how='inner')
m=m.merge(rsf_df,on='PatientID',how='inner')
e=m['Relapse'].astype(bool); t=m['RFS']
def cidx(col): return concordance_index_censored(e,t,m[col])[0]

# ---- 机械规则 ----
THR_STRENGTH=0.65   # 事先定: 成员单独OOF须>0.65
THR_CORR=0.50       # 事先定: 与已选成员相关须<0.5
print("="*60)
print(f"机械规则: 强度>{THR_STRENGTH}, 与已选成员Spearman相关<{THR_CORR}")
print("="*60)

# 1. 所有DL配置按强度排序
dl_names=list(pool.keys())
strengths={n:cidx(n) for n in dl_names}
ranked=sorted(dl_names,key=lambda n:strengths[n],reverse=True)

# 2. 贪心选择
selected=[]
for n in ranked:
    if strengths[n]<=THR_STRENGTH:
        continue
    # 与已选成员的最大相关
    if selected:
        corrs=[m[[n,s]].corr(method='spearman').iloc[0,1] for s in selected]
        maxc=max(corrs)
    else:
        maxc=0.0
    if maxc<THR_CORR:
        selected.append(n)
        print(f"  选中: {n:55s} OOF={strengths[n]:.4f} maxcorr={maxc:.3f}")
    else:
        print(f"  跳过: {n:55s} OOF={strengths[n]:.4f} maxcorr={maxc:.3f} (相关过高)")

print(f"\n机械规则选出 {len(selected)} 个成员:")
for s in selected: print(f"    {s}")

# 检查RSF是否入选
rsf_str=cidx('RSF')
print(f"\nRSF 强度={rsf_str:.4f} -> {'入选' if rsf_str>THR_STRENGTH else '被排除(弱于阈值)'}")

# ---- 融合结果 ----
for c in selected+['RSF']: m[c+'_r']=(m[c].rank()-1)/(len(m)-1)
def fuse(cols,weights=None):
    if weights is None: weights=[1/len(cols)]*len(cols)
    r=sum(w*m[c+'_r'] for c,w in zip(cols,weights))
    return concordance_index_censored(e,t,r)[0]

print("\n"+"="*60)
print("融合结果 (等权秩平均)")
print("="*60)
print(f"  机械规则成员等权:           {fuse(selected):.4f}")

# 短名映射(假设选出A/B/C)
print("\n  各组合对照(用短名):")
A='multitask/clinical+CT+PET+hard_T+hard_N'
B='task3/clinical+CT+PET'
C='task3/CT+PET+prob_T+prob_N+hard_T+hard_N'
combos={
 'A':[A],'B':[B],'C':[C],'RSF':['RSF'],
 'A+B':[A,B],'A+C':[A,C],'B+C':[B,C],'A+B+C':[A,B,C],
 'RSF+A':['RSF',A],'RSF+B':['RSF',B],'RSF+C':['RSF',C],
 'RSF+A+B':['RSF',A,B],'RSF+A+C':['RSF',A,C],'RSF+B+C':['RSF',B,C],
 'RSF+A+B+C':['RSF',A,B,C],
}
for name,cols in combos.items():
    try: print(f"    {name:12s}: {fuse(cols):.4f}")
    except Exception as ex: print(f"    {name:12s}: ERR")
