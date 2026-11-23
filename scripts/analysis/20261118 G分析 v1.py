# [20261118] G 分析：v3 ABLATION 0000 vs 1111；延遲（mean tracking time）、特徵點、ATE、觸發次數
import os,re,glob,random,subprocess,statistics as st
W=os.path.expanduser('~/slam_work'); ATE=f'{W}/scripts/ate_eval_local.py'
GT={'MH_01_easy':'MH_01','MH_02_easy':'MH_02','MH_03_medium':'MH_03','MH_04_difficult':'MH_04','MH_05_difficult':'MH_05','V2_03_difficult':'V_203'}
def perm(a,b,n=20000,seed=1):
    random.seed(seed); obs=st.median(a)-st.median(b); p=a+b; k=len(a); c=0
    for _ in range(n):
        random.shuffle(p)
        if abs(st.median(p[:k])-st.median(p[k:]))>=abs(obs)-1e-12: c+=1
    return obs,(c+1)/(n+1)
def load(t,s):
    R=[]
    for i in range(1,6):
        d=f'{W}/output/{t}/{s}'; log=open(f'{d}/run_{i}.log',errors='ignore').read()
        mt=float(re.search(r'mean tracking time\s*:?\s*([0-9.e-]+)',log,re.I).group(1))*1000
        kp=st.median(int(l.split()[2]) for l in open(f'{d}/frametimes_{i}.txt') if l.strip())
        o=subprocess.run(['python3','-I',ATE,f'{W}/result/ATE/{GT[s]}/euroc_groundtruth.txt',f'{d}/KeyFrameTrajectory_{i}.txt'],capture_output=True,text=True).stdout
        m=re.search(r'([0-9.]+) m\s+n=',o)
        R.append(dict(mt=mt,kp=kp,ate=float(m.group(1)) if m else None,emer=log.count('[EMERGENCY KF]'),fail=log.count('Fail to track local map')))
    return R
C={}
for s in GT:
    a,b=load('ABLV3_1111',s),load('ABLV3_0000',s)
    on=[r['mt'] for r in a]; off=[r['mt'] for r in b]; d,p=perm(on,off)
    ka,kb=st.median(r['kp'] for r in a),st.median(r['kp'] for r in b)
    C[s]=(st.median(on),st.median(off))
    ae=[r['ate'] for r in a if r['ate']]; be=[r['ate'] for r in b if r['ate']]
    print(f"{s}: 延遲 1111 {st.median(on):.2f}（{min(on):.2f}–{max(on):.2f}） 0000 {st.median(off):.2f}（{min(off):.2f}–{max(off):.2f}） 差 {d:+.2f} ms p={p:.4f} | kp {ka}/{kb}（差 {100*abs(ka-kb)/kb:.1f}%） | ATE {st.median(ae):.4f}/{st.median(be):.4f} | EMER {st.median(r['emer'] for r in a)} | trackfail {st.median(r['fail'] for r in a)}/{st.median(r['fail'] for r in b)}")
A=st.mean(v[0] for v in C.values()); B=st.mean(v[1] for v in C.values())
print(f'\n6 序列平均：1111 {A:.2f} 0000 {B:.2f} 差 {A-B:+.2f} ms（{100*(A-B)/B:+.1f}%）')
hard=st.mean(C[s][0]-C[s][1] for s in ['MH_04_difficult','MH_05_difficult','V2_03_difficult']); easy=st.mean(C[s][0]-C[s][1] for s in ['MH_01_easy','MH_02_easy','MH_03_medium'])
print(f'G3：困難序列平均成本 {hard:+.2f} ms，簡單 {easy:+.2f} ms，差 {hard-easy:+.2f} ms')
