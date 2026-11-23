# [20261122] L1–L4 分析：依 20261121 事先登記 補實驗L1-L4 v2.md（判定條件不改）
# 用法：python3 "20261122 L分析 v1.py" <L1|L2|L3|L4>
# 追蹤比例＝frametimes 第 5 欄==2 ÷ 全部幀；追蹤失敗次數＝逐幀狀態從 2 變成非 2 的次數（三個方法同一算法，不靠 log）
# 初始化成功＝該次 run 至少有一幀 state==2；崩潰（無 frametimes 或無 Mean tracking time）＝主分析排除，敏感度當最差（追蹤比例 0）
# ATE：Sim3；只在估計軌跡的時間跨度 ≥ 序列時間跨度 30% 時報，否則「不可比」
import os, re, sys, random, subprocess, statistics as st, importlib.util
import numpy as np
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
B4 = os.path.join(ROOT, '4 第二次投稿'); RAW = os.path.join(B4, '2 實驗結果', '20261122 L1-L4原始輸出 v1')
GTD = os.path.join(B4, '2 實驗結果', '20261118 EuRoC groundtruth v1'); ATEP = os.path.join(B4, '1 實驗腳本', '20261115 ATE評估 v1.py')
ADV = os.path.expanduser('~/slam_work/datasets/ADVIO')
spec = importlib.util.spec_from_file_location('atem', ATEP); A = importlib.util.module_from_spec(spec); spec.loader.exec_module(A)
GT = {'MH_01_easy': 'MH_01', 'MH_02_easy': 'MH_02', 'MH_03_medium': 'MH_03', 'MH_04_difficult': 'MH_04', 'MH_05_difficult': 'MH_05', 'V2_03_difficult': 'V_203', 'V2_02_medium': 'V_202'}

def perm(a, b, n=20000, seed=1):
    random.seed(seed); obs = st.median(a) - st.median(b); p = list(a) + list(b); k = len(a); c = 0
    for _ in range(n):
        random.shuffle(p)
        if abs(st.median(p[:k]) - st.median(p[k:])) >= abs(obs) - 1e-12: c += 1
    return obs, (c + 1) / (n + 1)

def rng(v, f='.1f'): return '—' if not v else f'{st.median(v):{f}}（{min(v):{f}}–{max(v):{f}}，n={len(v)}）'

def ate_advio(seq, kf, nframes, T):
    if not os.path.exists(kf) or os.path.getsize(kf) == 0: return None, 'noTraj'
    e = A.load_tum(kf)
    if len(e) < 3: return None, 'few'
    gt = A.load_tum(f'{ADV}/{seq}/ground-truth/groundtruth_tum.txt')
    cov = (e[-1, 0] - e[0, 0]) / (T[-1] - T[0])
    if cov < 0.30: return None, f'不可比（覆蓋 {cov*100:.0f}%）'
    gg, ee = A.associate(gt, e)
    if len(gg) < 3: return None, 'few'
    s, R, t = A.umeyama_sim3(ee, gg); err = np.linalg.norm((s * (R @ ee.T)).T + t - gg, axis=1)
    return float(np.sqrt((err ** 2).mean())), f'cov{cov*100:.0f}%'

def ate_euroc(seq, kf, tspan):
    o = subprocess.run([sys.executable, '-I', '-X', 'utf8', ATEP, os.path.join(GTD, GT[seq], 'euroc_groundtruth.txt'), kf], capture_output=True, text=True, encoding='utf-8').stdout
    a = re.search(r'([0-9.]+) m\s+n=', o); return float(a.group(1)) if a else None

def load_run(tag, seq, i, kind):
    d = os.path.join(RAW, tag, seq); lp = os.path.join(d, f'run_{i}.log'); ft = os.path.join(d, f'frametimes_{i}.txt'); kf = os.path.join(d, f'KeyFrameTrajectory_{i}.txt')
    if not os.path.exists(lp): return dict(i=i, crash=True, why='no log')
    log = open(lp, errors='ignore').read(); m = re.search(r'mean tracking time\s*:?\s*([0-9.e-]+)', log, re.I)
    if not m or not os.path.exists(ft): return dict(i=i, crash=True, why='無 Mean tracking time 或 frametimes')
    L = [l.split() for l in open(ft) if l.strip()]
    S = [x[4] for x in L]; ok = sum(1 for s in S if s == '2'); fail = sum(1 for k in range(1, len(S)) if S[k - 1] == '2' and S[k] != '2')
    first = next((k for k, s in enumerate(S) if s == '2'), None)
    feats = [int(x[2]) for x in L if int(x[2]) > 0]
    r = dict(i=i, crash=False, mt=float(m.group(1)) * 1000, ok=100 * ok / len(S), fail=fail, init=first is not None, first=first,
             feat=st.median(feats) if feats else None, maps=log.count('Creation of new map with id'), n=len(S))
    if kind == 'advio':
        T = [float(l.split(',')[0]) for l in open(f'{ADV}/{seq}/iphone/frames.csv') if l.strip()]
        r['ate'], r['atenote'] = ate_advio(seq, kf, len(S), T)
    elif kind == 'euroc' and os.path.exists(kf): r['ate'] = ate_euroc(seq, kf, None)
    cf = os.path.join(d, f'cand_{i}.txt')
    if os.path.exists(cf):
        C = np.loadtxt(cf, ndmin=2); r['cand'] = float(np.median(C[:, 0])); r['ext'] = float(np.median(C[:, 4])); r['det'] = float(np.median(C[:, 1])); r['qt'] = float(np.median(C[:, 3]))
    return r

def ok_(R, k): return [r[k] for r in R if not r['crash'] and r.get(k) is not None]

W = sys.argv[1]
if W == 'L3':
    print('== L3 記憶體（EuRoC MH04，每方法 5 次；GPU＝nvidia-smi 依行程 PID，host＝VmHWM（行程峰值 RSS），MiB）')
    R = {}
    for m in ('final', 'b20', 'jslam'):
        for i in range(1, 6):
            L = [l.split() for l in open(os.path.join(RAW, 'L3_MEM', f'mem_{m}_{i}.txt')) if l.strip()]
            R.setdefault(m, []).append((max(float(x[4]) for x in L), max(float(x[3]) for x in L) / 1024))
        g = [a for a, _ in R[m]]; h = [b for _, b in R[m]]
        print(f'  {m:6s} GPU {rng(g,".0f")}  host {rng(h,".1f")}')
    for m in ('final', 'b20', 'jslam'):
        runs = [load_run(f'L3_{m}', 'MH_04_difficult', i, 'none') for i in range(1, 6)]
        print(f'  {m:6s} 崩潰 {sum(r["crash"] for r in runs)} 次；延遲 ms {rng(ok_(runs,"mt"),".2f")}')
    dg = st.median([a for a, _ in R['final']]) - st.median([a for a, _ in R['b20']]); dh = st.median([b for _, b in R['final']]) - st.median([b for _, b in R['b20']])
    print(f'  P3a GPU 增量（Final−Baseline-20，中位）{dg:+.1f} MiB ≤ 10 → {"成立" if dg <= 10 else "不成立"}')
    print(f'  P3b host 增量 {dh:+.1f} MiB ≤ 20 → {"成立" if dh <= 20 else "不成立"}')
    print(f'  （附帶）Final 對 Jetson-SLAM：GPU {st.median([a for a,_ in R["final"]])-st.median([a for a,_ in R["jslam"]]):+.0f} MiB，host {st.median([b for _,b in R["final"]])-st.median([b for _,b in R["jslam"]]):+.1f} MiB')

elif W == 'L1':
    SEQ = ['advio-01', 'advio-02', 'advio-03', 'advio-04']; M = ['final', 'b20', 'jslam', 'orb3']; D = {}
    for s in SEQ:
        for m in M: D[m, s] = [load_run(f'L1_{m}', s, i, 'advio') for i in range(1, 6)]
    print('== L1 逐序列逐方法（追蹤比例%、追蹤失敗次數、初始化成功次數、第一個追蹤幀、延遲 ms、特徵數、ATE、ORB3 地圖數）')
    for s in SEQ:
        print(f'-- {s}（{D["final",s][0]["n"] if not D["final",s][0]["crash"] else "?"} 幀）')
        for m in M:
            R = D[m, s]; cr = [r['i'] for r in R if r['crash']]
            print(f'  {m:6s} 追蹤比例 {rng(ok_(R,"ok"))} | 失敗 {[r["fail"] for r in R if not r["crash"]]} | 初始化成功 {sum(1 for r in R if not r["crash"] and r["init"])}/{len(R)-len(cr)} | 首追蹤幀 {sorted(r["first"] for r in R if not r["crash"] and r["first"] is not None)}'
                  f' | 延遲 {rng(ok_(R,"mt"),".2f")} | 特徵數中位 {rng(ok_(R,"feat"),".0f")} | ATE {[ (round(r["ate"],3) if r.get("ate") is not None else r.get("atenote")) for r in R if not r["crash"]]}'
                  + (f' | 地圖數 {[r["maps"] for r in R if not r["crash"]]}' if m == 'orb3' else '') + (f' | 崩潰 run {cr}' if cr else ''))
    med = lambda m, s: st.median(ok_(D[m, s], 'ok'))
    a = [s for s in SEQ if med('final', s) - med('b20', s) > 0]; ge = all(med('final', s) >= med('b20', s) for s in SEQ)
    print(f'\n== P1a：Final ≥ Baseline-20 且至少 3/4 序列差距 > 0：{len(a)}/4 序列 > 0；全部 ≥：{ge} → {"成立" if len(a) >= 3 and ge else "不成立"}')
    print('  逐序列 Final−b20（個百分點）：' + '、'.join(f'{s} {med("final",s)-med("b20",s):+.1f}' for s in SEQ))
    b = [s for s in SEQ if med('final', s) > med('jslam', s)]; print(f'== P1b：Final > Jetson-SLAM，4/4：{len(b)}/4 → {"成立" if len(b)==4 else "不成立"}  ' + '、'.join(f'{s} {med("final",s)-med("jslam",s):+.1f}' for s in SEQ))
    c = [s for s in SEQ if med('orb3', s) > med('final', s)]; print(f'== P1c：ORB-SLAM3 > Final，4/4：{len(c)}/4 → {"成立" if len(c)==4 else "不成立"}  ' + '、'.join(f'{s} {med("orb3",s)-med("final",s):+.1f}' for s in SEQ))
    d = [s for s in SEQ if st.median(ok_(D['final', s], 'feat')) < st.median(ok_(D['jslam', s], 'feat'))]
    print(f'== P1d：Final 每幀特徵數 < Jetson-SLAM（逐序列）：{len(d)}/4 → {"成立" if len(d)==4 else "不成立"}  ' + '、'.join(f'{s} {st.median(ok_(D["final",s],"feat")):.0f} vs {st.median(ok_(D["jslam",s],"feat")):.0f}' for s in SEQ))
    print('\n== 檢定（追蹤比例，雙側排列檢定中位數差，20000 次；Bonferroni 0.05/8 = 0.00625）')
    for s in SEQ:
        for o in ('b20', 'jslam'):
            dd, p = perm(ok_(D['final', s], 'ok'), ok_(D[o, s], 'ok')); print(f'  {s} Final vs {o}: 差 {dd:+.1f}  p={p:.5f} {"✓" if p < 0.00625 else ""}')
    print('\n== 敏感度（崩潰當追蹤比例 0）：有崩潰的方法×序列')
    for (m, s), R in D.items():
        cr = [r for r in R if r['crash']]
        if cr: print(f'  {m} {s} 崩潰 {len(cr)} 次（{[r["why"] for r in cr]}）；補 0 後追蹤比例中位數 {st.median([r["ok"] if not r["crash"] else 0 for r in R]):.1f}（原 {st.median(ok_(R,"ok")):.1f}）')

elif W == 'L2':
    M = ['final', 'g10', 'b20', 'jslam', 'orb3']; G = [(c, s) for c in ('RC', 'RB') for s in ('MH_03_medium', 'V2_02_medium')]; D = {}
    for c, s in G:
        for m in M: D[m, c, s] = [load_run(f'L2_{c}_{m}', s, i, 'euroc') for i in range(1, 6)]
    print('== L2 區域退化（追蹤比例%、ATE、失敗次數、延遲 ms、特徵數；ORB3 地圖數）')
    for c, s in G:
        print(f'-- {c} {s}')
        for m in M:
            R = D[m, c, s]; cr = [r['i'] for r in R if r['crash']]
            print(f'  {m:6s} 追蹤比例 {rng(ok_(R,"ok"))} | ATE {rng(ok_(R,"ate"),".3f")} | 失敗 {[r["fail"] for r in R if not r["crash"]]} | 延遲 {rng(ok_(R,"mt"),".2f")} | 特徵 {rng(ok_(R,"feat"),".0f")} | 初始化 {sum(1 for r in R if not r["crash"] and r["init"])}/{len(R)-len(cr)}'
                  + (f' | 地圖 {[r["maps"] for r in R if not r["crash"]]}' if m == 'orb3' else '') + (f' | 崩潰 {cr}' if cr else ''))
    med = lambda m, c, s, k='ok': st.median(ok_(D[m, c, s], k))
    p2a = [(c, s) for c, s in G if med('final', c, s) - med('b20', c, s) >= 5]
    print(f'\n== P2a：Final ≥ Baseline-20 ＋ 5 個百分點，至少 3/4 組：{len(p2a)}/4 → {"成立" if len(p2a)>=3 else "不成立"}  ' + '、'.join(f'{c}-{s[:4]} {med("final",c,s)-med("b20",c,s):+.1f}' for c, s in G))
    p2b = [(c, s) for c, s in G if abs(med('final', c, s) - med('g10', c, s)) <= 5]
    print(f'== P2b：Final 與 G10 追蹤比例差在 ±5 內（逐組）：{len(p2b)}/4 → {"成立（4/4）" if len(p2b)==4 else "不成立（描述）"}  ' + '、'.join(f'{c}-{s[:4]} {med("final",c,s)-med("g10",c,s):+.1f}' for c, s in G))
    p2c = [(c, s) for c, s in G if med('final', c, s, 'mt') <= med('g10', c, s, 'mt')]
    print(f'== P2c：Final 中位延遲不高於 G10（逐組）：{len(p2c)}/4 → {"成立（4/4）" if len(p2c)==4 else "不成立（描述）"}  ' + '、'.join(f'{c}-{s[:4]} {med("final",c,s,"mt")-med("g10",c,s,"mt"):+.2f}ms' for c, s in G))
    print('\n== 檢定（追蹤比例；Bonferroni 0.05/8 = 0.00625）')
    for c, s in G:
        for o in ('b20', 'g10'):
            dd, p = perm(ok_(D['final', c, s], 'ok'), ok_(D[o, c, s], 'ok')); print(f'  {c} {s} Final vs {o}: 差 {dd:+.1f} p={p:.5f} {"✓" if p < 0.00625 else ""}')
    print('\n== 敏感度（崩潰當 0）')
    for k, R in D.items():
        cr = [r for r in R if r['crash']]
        if cr: print(f'  {k} 崩潰 {len(cr)} 次；補 0 後追蹤比例中位數 {st.median([r["ok"] if not r["crash"] else 0 for r in R]):.1f}')

elif W == 'L4':
    SC = [('S0', 'MH_01_easy'), ('S0', 'MH_02_easy'), ('S0', 'MH_03_medium'), ('S0', 'MH_04_difficult'), ('S0', 'MH_05_difficult'), ('S0', 'V2_03_difficult'), ('S0', 'V2_02_medium'), ('C25', 'MH_03_medium'), ('C25', 'V2_02_medium')]
    D = {}; ra = []; rb = []; rc = []
    print('== L4（候選點數＝NMS 後四元樹前 8 層加總的每幀中位；擷取延遲＝total_extraction 每幀中位 ms；總延遲＝log Mean tracking time ms）')
    for c, s in SC:
        for m in ('final', 'g10'): D[m, c, s] = [load_run(f'L4_{c}_{m}', s, i, 'euroc') for i in range(1, 6)]
        F, G10 = D['final', c, s], D['g10', c, s]
        cf, cg = ok_(F, 'cand'), ok_(G10, 'cand'); ef, eg = ok_(F, 'ext'), ok_(G10, 'ext'); tf, tg = ok_(F, 'mt'), ok_(G10, 'mt')
        ratio = st.median(cg) / st.median(cf); ra.append(ratio); rb.append(st.median(eg) - st.median(ef)); rc.append(st.median(tg) - st.median(tf))
        print(f'-- {c} {s}\n   候選點 Final {rng(cf,".0f")}  G10 {rng(cg,".0f")}  比值 {ratio:.2f}\n   擷取延遲 Final {rng(ef,".2f")}  G10 {rng(eg,".2f")}  差 {rb[-1]:+.2f}\n   總延遲 Final {rng(tf,".2f")}  G10 {rng(tg,".2f")}  差 {rc[-1]:+.2f}'
              f'\n   失敗 Final {[r["fail"] for r in F if not r["crash"]]}  G10 {[r["fail"] for r in G10 if not r["crash"]]} | 追蹤比例 Final {rng(ok_(F,"ok"))}  G10 {rng(ok_(G10,"ok"))} | 保留特徵 Final {rng(ok_(F,"feat"),".0f")}  G10 {rng(ok_(G10,"feat"),".0f")}'
              + (f'\n   崩潰：{[(m, r["i"]) for m in ("final","g10") for r in D[m,c,s] if r["crash"]]}' if any(r['crash'] for m in ('final', 'g10') for r in D[m, c, s]) else ''))
    print(f'\n== P4a：G10 候選點數 ≥ Final 的 1.5 倍（中位）：{sum(x>=1.5 for x in ra)}/9 場景；比值中位 {st.median(ra):.2f}（{min(ra):.2f}–{max(ra):.2f}） → {"成立" if all(x>=1.5 for x in ra) else "不成立（逐場景見上）"}')
    print(f'== P4b：G10 擷取延遲 > Final：{sum(x>0 for x in rb)}/9 場景；差中位 {st.median(rb):+.2f} ms → {"成立" if all(x>0 for x in rb) else "不成立（逐場景見上）"}')
    print(f'== P4c：兩者總延遲差 < 0.6 ms（絕對值）：{sum(abs(x)<0.6 for x in rc)}/9 場景；差中位 {st.median(rc):+.2f} ms（{min(rc):+.2f}～{max(rc):+.2f}） → {"成立" if all(abs(x)<0.6 for x in rc) else "不成立（逐場景見上）"}')
