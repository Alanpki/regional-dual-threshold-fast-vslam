# [20261116] ADVIO 彙整：整段＋前 3500 幀（先前實作者的範圍）分開算
# 用法：python3 summarize_advio.py <ATE評估.py> <TAG> [TAG...]
# 每次執行：OK 幀比例（frametimes state==2）、第一次 OK 的幀、第一次追丟（OK 之後 state==3）的幀、
#           ATE（Sim3，KeyFrameTrajectory 對 GT；只取 t ≤ 該範圍最後一幀的時間）、對到的 KF 數、新地圖數（ORB-SLAM3）、
#           EMER／AdaptNN 觸發次數、mean 延遲
import sys, os, re, glob, statistics as st, importlib.util
import numpy as np
W = os.path.expanduser('~/slam_work')
spec = importlib.util.spec_from_file_location('atem', sys.argv[1]); A = importlib.util.module_from_spec(spec); spec.loader.exec_module(A)
TAGS = sys.argv[2:]

def ate_cut(gt, tr, tmax):
    if not os.path.exists(tr) or os.path.getsize(tr) == 0:
        return None, 0
    e = A.load_tum(tr); e = e[e[:, 0] <= tmax]
    g = gt[gt[:, 0] <= tmax + 0.05]
    if len(e) < 3:
        return None, len(e)
    gg, ee = A.associate(g, e)
    if len(gg) < 3:
        return None, len(gg)
    s, R, t = A.umeyama_sim3(ee, gg)
    err = np.linalg.norm((s * (R @ ee.T)).T + t - gg, axis=1)
    return float(np.sqrt((err ** 2).mean())), len(gg)

def stats(L):
    ok = [x[4] == '2' for x in L]
    first_ok = next((k + 1 for k, v in enumerate(ok) if v), None)
    first_lost = None
    if first_ok:
        first_lost = next((k + 1 for k in range(first_ok, len(L)) if L[k][4] == '3'), None)
    return sum(ok) / len(L) * 100, first_ok, first_lost

f = lambda v, n=3: '-' if v is None else f'{v:.{n}f}'
for tag in TAGS:
    for od in sorted(glob.glob(f'{W}/output/{tag}/*/')):
        seq = os.path.basename(od.rstrip('/'))
        gt = A.load_tum(f'{W}/datasets/ADVIO/{seq}/ground-truth/groundtruth_tum.txt')
        rows = []
        for i in range(1, 6):
            lp = f'{od}/run_{i}.log'
            if not os.path.exists(lp):
                continue
            log = open(lp, errors='ignore').read()
            mt = re.search(r'mean tracking time\s*:?\s*([0-9.e-]+)', log, re.I)
            L = [l.split() for l in open(f'{od}/frametimes_{i}.txt') if l.strip()] if os.path.exists(f'{od}/frametimes_{i}.txt') else []
            # frametimes 的時間戳精度不足（setprecision 9 仍可，但以 csv 為準）→ 時間取 frames.csv
            csvp = f'{W}/datasets/ADVIO/{seq}/iphone/frames.csv'
            T = [float(r.split(',')[0]) for r in open(csvp) if r.strip()]
            r = dict(i=i, mean=float(mt.group(1)) * 1000 if mt else None, n=len(L),
                     emer=log.count('[EMERGENCY KF]'), ann=log.count('[Adaptive nn]'),
                     maps=len(re.findall(r'Creation of new map with id', log)), reloc_fail=log.count('Fail to relocalization'))
            kf = f'{od}/KeyFrameTrajectory_{i}.txt'
            for name, N in (('all', len(L)), ('3500', min(3500, len(L)))):
                if N == 0:
                    r[name] = (None, None, None, None, 0); continue
                okp, fo, fl = stats(L[:N])
                a, n = ate_cut(gt, kf, T[N - 1])
                r[name] = (okp, fo, fl, a, n)
            rows.append(r)
        if not rows:
            continue
        print(f'== {tag} {seq}（{rows[0]["n"]} 幀）')
        for r in rows:
            s = f"  run{r['i']} mean={f(r['mean'],2)}ms EMER={r['emer']} AdaptNN={r['ann']} maps={r['maps']} relocFail={r['reloc_fail']}"
            for name in ('all', '3500'):
                okp, fo, fl, a, n = r[name]
                s += f" | {name}: OK={f(okp,1)}% 初始化@{fo} 追丟@{fl} ATE={f(a)}(n={n})"
            print(s)
        for name in ('all', '3500'):
            oks = [r[name][0] for r in rows if r[name][0] is not None]
            ates = [r[name][3] for r in rows if r[name][3] is not None]
            fos = [r[name][1] for r in rows if r[name][1]]
            fls = [r[name][2] for r in rows if r[name][2]]
            print(f"  -> {name}: OK 幀中位數 {f(st.median(oks),1)}%（{f(min(oks),1)}–{f(max(oks),1)}）; 初始化幀 {sorted(fos)}; 追丟幀 {sorted(fls)}; "
                  f"ATE 中位數 {f(st.median(ates)) if ates else '-'}（{len(ates)}/{len(rows)} 有值）")
        print(f"  -> mean 延遲中位數 {f(st.median([r['mean'] for r in rows if r['mean']]),2)} ms")
