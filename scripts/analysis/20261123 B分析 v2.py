# [20261123] B 分析 v2：舊格式 frametimes（M1）只比延遲與 ATE。
# [20261122] B 分析：JM（Jetson-SLAM＋對齊初始化門檻）對原版 Jetson-SLAM，依 20261122 事先登記 Jetson-SLAM初始化門檻對齊重跑 v1.md
# 用法：python3 "20261122 B分析 v1.py" <B1|B2|B3|B4|B5|B6|ALL> [noate]
# 指標與規則同 L分析 v1.py（追蹤比例＝state==2 ÷ 全部；失敗＝由 2 轉出；初始化成功＝至少一幀 state==2；崩潰排除、敏感度補 0）
import os, re, sys, statistics as st, subprocess, importlib.util
import numpy as np
O = os.path.expanduser('~/slam_work/output'); ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
B4 = os.path.join(ROOT, '4 第二次投稿'); GTD = os.path.join(B4, '2 實驗結果', '20261118 EuRoC groundtruth v1'); ATEP = os.path.join(B4, '1 實驗腳本', '20261115 ATE評估 v1.py')
ADV = os.path.expanduser('~/slam_work/datasets/ADVIO'); NOATE = 'noate' in sys.argv
spec = importlib.util.spec_from_file_location('atem', ATEP); A = importlib.util.module_from_spec(spec); spec.loader.exec_module(A)
GT = {'MH_01_easy': 'MH_01', 'MH_02_easy': 'MH_02', 'MH_03_medium': 'MH_03', 'MH_04_difficult': 'MH_04', 'MH_05_difficult': 'MH_05', 'V2_02_medium': 'V_202'}

def ate_euroc(seq, kf):
    if NOATE or not os.path.exists(kf): return None
    o = subprocess.run([sys.executable, '-I', '-X', 'utf8', ATEP, os.path.join(GTD, GT[seq], 'euroc_groundtruth.txt'), kf], capture_output=True, text=True, encoding='utf-8').stdout
    a = re.search(r'([0-9.]+) m\s+n=', o); return float(a.group(1)) if a else None
def ate_advio(seq, kf, T):
    if NOATE or not os.path.exists(kf) or os.path.getsize(kf) == 0: return None
    e = A.load_tum(kf)
    if len(e) < 3 or (e[-1, 0] - e[0, 0]) / (T[-1] - T[0]) < 0.30: return None
    gt = A.load_tum(f'{ADV}/{seq}/ground-truth/groundtruth_tum.txt'); gg, ee = A.associate(gt, e)
    if len(gg) < 3: return None
    s, R, t = A.umeyama_sim3(ee, gg); return float(np.sqrt((np.linalg.norm((s * (R @ ee.T)).T + t - gg, axis=1) ** 2).mean()))
def load(tag, seq, kind, runs=range(1, 6)):
    R = []
    for i in runs:
        d = os.path.join(O, tag, seq); lp = f'{d}/run_{i}.log'; ft = f'{d}/frametimes_{i}.txt'
        if not os.path.exists(ft): continue
        log = open(lp, errors='ignore').read() if os.path.exists(lp) else ''
        m = re.search(r'mean tracking time\s*:?\s*([0-9.e-]+)', log, re.I)
        if not m: R.append(dict(i=i, crash=True)); continue
        L = [l.split() for l in open(ft) if l.strip()]
        if L and len(L[0]) < 5:  # 舊格式（M1 批，沒有狀態欄）：只比延遲與 ATE
            kf = f'{d}/KeyFrameTrajectory_{i}.txt'; R.append(dict(i=i, crash=False, mt=float(m.group(1)) * 1000, ok=None, fail=None, init=True, feat=None, ate=ate_euroc(seq, kf) if kind == 'euroc' else None)); continue
        S = [x[4] for x in L]
        feats = [int(x[2]) for x in L if int(x[2]) > 0]; kf = f'{d}/KeyFrameTrajectory_{i}.txt'
        r = dict(i=i, crash=False, mt=float(m.group(1)) * 1000, ok=100 * sum(s == '2' for s in S) / len(S), fail=sum(1 for k in range(1, len(S)) if S[k - 1] == '2' and S[k] != '2'),
                 init=any(s == '2' for s in S), feat=st.median(feats) if feats else None)
        if kind == 'euroc': r['ate'] = ate_euroc(seq, kf)
        else:
            T = [float(l.split(',')[0]) for l in open(f'{ADV}/{seq}/iphone/frames.csv') if l.strip()]; r['ate'] = ate_advio(seq, kf, T)
        R.append(r)
    return R
def ok_(R, k): return [r[k] for r in R if not r['crash'] and r.get(k) is not None]
def rng(v, f='.1f'): return '—' if not v else f'{st.median(v):{f}}（{min(v):{f}}–{max(v):{f}}，n={len(v)}）'
def initn(R): return f'{sum(1 for r in R if not r["crash"] and r["init"])}/{sum(1 for r in R if not r["crash"])}'
FLAGS = []
def cell(name, orig, jm, kind):
    print(f'-- {name}')
    for lab, R in (('原版', orig), ('JM  ', jm)):
        print(f'  {lab} 追蹤比例 {rng(ok_(R,"ok"))} | 失敗 {[r["fail"] for r in R if not r["crash"] and r["fail"] is not None]} | 初始化 {initn(R)} | 延遲 {rng(ok_(R,"mt"),".2f")} | 特徵 {rng(ok_(R,"feat"),".0f")}'
              + (f' | ATE {rng(ok_(R,"ate"),".3f")}' if any(r.get('ate') is not None for r in R) else '') + (f' | 崩潰 {[r["i"] for r in R if r["crash"]]}' if any(r["crash"] for r in R) else ''))
    if not ok_(orig, 'ok') and ok_(jm, 'mt'):
        print(f'  （原版為舊格式，無狀態欄；只比較延遲與 ATE）延遲差 {st.median(ok_(jm,"mt"))-st.median(ok_(orig,"mt")):+.2f} ms')
    if ok_(orig, 'ok') and ok_(jm, 'ok'):
        d = st.median(ok_(jm, 'ok')) - st.median(ok_(orig, 'ok')); di = initn(orig) != initn(jm)
        flag = abs(d) > 5 or di; FLAGS.append((name, flag)); print(f'  差（JM−原版）追蹤比例 {d:+.1f} 點；延遲 {st.median(ok_(jm,"mt"))-st.median(ok_(orig,"mt")):+.2f} ms；初始化次數{"不同" if di else "相同"} → {"有差異（>5 點或初始化不同）" if flag else "無差異"}')

W = sys.argv[1]
SEQ2 = ['MH_03_medium', 'V2_02_medium']
if W in ('B1', 'ALL'):
    print('== B1 Fig. 2（對比降低）')
    for C in ('S0', 'C50', 'C35', 'C25'):
        for s in SEQ2: cell(f'{C} {s}', load(f'DEG_{C}_base', s, 'euroc'), load(f'JM_B1_{C}', s, 'euroc'), 'euroc')
if W in ('B2', 'ALL'):
    print('\n== B2 ADVIO')
    for s in ('advio-01', 'advio-02', 'advio-03', 'advio-04'): cell(s, load('L1_jslam', s, 'advio'), load('JM_B2', s, 'advio'), 'advio')
if W in ('B3', 'ALL'):
    print('\n== B3 Table I（EuRoC MH01–05）')
    for s in ('MH_01_easy', 'MH_02_easy', 'MH_03_medium', 'MH_04_difficult', 'MH_05_difficult'): cell(s, load('M1_baseline', s, 'euroc'), load('JM_B3', s, 'euroc'), 'euroc')
    for lab, tag in (('原版', 'M1_baseline'), ('JM', 'JM_B3')):
        ms = [st.median(ok_(load(tag, s, 'euroc'), 'mt')) for s in ('MH_01_easy', 'MH_02_easy', 'MH_03_medium', 'MH_04_difficult', 'MH_05_difficult')]
        print(f'  {lab} 五序列平均延遲（各序列中位數的平均）{st.mean(ms):.2f} ms')
if W in ('B4', 'ALL'):
    print('\n== B4 區域退化')
    for C in ('RC', 'RB'):
        for s in SEQ2: cell(f'{C} {s}', load(f'L2_{C}_jslam', s, 'euroc'), load(f'JM_B4_{C}', s, 'euroc'), 'euroc')
if W in ('B5', 'ALL'):
    print('\n== B5 記憶體（MH04，MiB；GPU＝nvidia-smi 依 PID，host＝VmHWM）')
    for lab, d, pre in (('原版', f'{O}/L3_MEM', 'mem_jslam_'), ('JM', f'{O}/JM_B5_MEM', 'mem_')):
        g = []; h = []
        for i in range(1, 6):
            f = f'{d}/{pre}{i}.txt'
            if not os.path.exists(f): continue
            L = [l.split() for l in open(f) if l.strip()]
            if L: g.append(max(float(x[4]) for x in L)); h.append(max(float(x[3]) for x in L) / 1024)
        print(f'  {lab} GPU {rng(g,".0f")} host {rng(h,".1f")}')
if W in ('B6', 'ALL'):
    print('\n== B6 48 像素網格（JM；每序列 3 次）')
    for s in ('MH_01_easy', 'MH_03_medium', 'V2_02_medium'):
        R = load('JM_B6', s, 'euroc', range(1, 4)); print(f'  {s}: 追蹤比例 {rng(ok_(R,"ok"))} 初始化 {initn(R)} 特徵 {rng(ok_(R,"feat"),".0f")} 崩潰 {[r["i"] for r in R if r["crash"]]}')
if FLAGS:
    n = sum(1 for _, f in FLAGS if f)
    print(f'\n== 決策規則：{n}/{len(FLAGS)} 格有差異（>5 個百分點或初始化次數不同） → ' + ('全面改用 JM 並寫明初始化門檻對齊' if n else '維持現有數字，Section IV-A 加一句已核對'))
    print('   有差異的格：' + '、'.join(nm for nm, f in FLAGS if f))
