# [20261118] A 影像退化掃描分析：OK 幀比例（frametimes 第 5 欄 ==2 ÷ 全部幀）；置換檢定（中位數差，20000 次，雙尾）；Bonferroni 0.0125
import os, random, statistics as st
W=os.path.expanduser('~/slam_work/output')
SEQ=['MH_03_medium','V2_02_medium']; COND=['S0','C50','C35','C25','K2','K3','K4']; M=['ours1000','ours0000','base']
def ok(c,m,s):
    R=[]
    for i in range(1,6):
        f=f'{W}/DEG_{c}_{m}/{s}/frametimes_{i}.txt'
        L=[l.split() for l in open(f) if l.strip()] if os.path.exists(f) else []
        R.append(100*sum(1 for x in L if x[4]=='2')/len(L) if L else 0.0)
    return R
def fails(c,m,s):
    return [open(f'{W}/DEG_{c}_{m}/{s}/run_{i}.log',errors='ignore').read().count('Fail to track local map') for i in range(1,6)]
def perm(a,b,n=20000,seed=1):
    random.seed(seed); obs=st.median(a)-st.median(b); p=a+b; k=len(a); c=0
    for _ in range(n):
        random.shuffle(p)
        if abs(st.median(p[:k])-st.median(p[k:]))>=abs(obs)-1e-12: c+=1
    return obs,(c+1)/(n+1)
D={(c,m,s):ok(c,m,s) for c in COND for m in M for s in SEQ}
print('== OK 幀比例 %：中位數（最小–最大），每格 5 次')
for s in SEQ:
    print(f'-- {s}')
    for c in COND:
        print(f'  {c:4s} '+'  '.join(f'{m}: {st.median(D[c,m,s]):5.1f}（{min(D[c,m,s]):.1f}–{max(D[c,m,s]):.1f}）' for m in M)
                 +'  | trackfail 中位數 '+' '.join(f'{m}={st.median(fails(c,m,s))}' for m in M))
TH=0.05/4
print(f'\n== 檢定（門檻 {TH}）')
for s in SEQ:
    for grp,cs in [('低對比',['C50','C35','C25']),('跳幀',['K2','K3','K4'])]:
        for a_m,b_m,lab in [('ours1000','ours0000','A1/A3 開 vs 關'),('ours1000','base','A4 本方法 vs Jetson-SLAM')]:
            if grp=='跳幀' and a_m!='ours1000' or (grp=='跳幀' and b_m=='base'): pass
            a=sum((D[c,a_m,s] for c in cs),[]); b=sum((D[c,b_m,s] for c in cs),[])
            d,p=perm(a,b); print(f'  {s} {grp} {lab}: 中位數 {st.median(a):.1f} vs {st.median(b):.1f} 差={d:+.1f} p={p:.4f} {"顯著" if p<TH else ""}')
print('\n== A2：開−關 OK 幀中位數差')
for s in SEQ:
    print(f'  {s}: '+'  '.join(f'{c}={st.median(D[c,"ours1000",s])-st.median(D[c,"ours0000",s]):+.1f}' for c in COND))
