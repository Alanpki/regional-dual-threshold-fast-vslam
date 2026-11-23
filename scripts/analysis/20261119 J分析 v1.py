# [20261119] J 分析：依 20261118 事先登記 最終系統1000與低門檻對照 v1.md（判定條件不改）
# 指標：延遲＝run log Mean tracking time；ATE＝Sim3（20261115 ATE評估 v1.py）；追蹤失敗＝log「Fail to track local map」；
#       OK 幀比例＝frametimes 第 5 欄 ==2 ÷ 全部幀；初始化幀＝第一個狀態 2 的列號
# 閃退（無軌跡或無 Mean tracking time）：主分析排除；敏感度當最差（比照 11-17 補記 2）
import os, re, random, subprocess, sys, statistics as st

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
B4 = os.path.join(ROOT, '4 第二次投稿')
RAW = os.path.join(B4, '2 實驗結果', '20261119 J原始輸出 v1')
GTD = os.path.join(B4, '2 實驗結果', '20261118 EuRoC groundtruth v1')
ATE = os.path.join(B4, '1 實驗腳本', '20261115 ATE評估 v1.py')
GT = {'MH_01_easy': 'MH_01', 'MH_02_easy': 'MH_02', 'MH_03_medium': 'MH_03', 'MH_04_difficult': 'MH_04',
      'MH_05_difficult': 'MH_05', 'V2_03_difficult': 'V_203', 'V2_02_medium': 'V_202'}
CACHE = {}


def perm(a, b, n=20000, seed=1):
    random.seed(seed); obs = st.median(a) - st.median(b); p = a + b; k = len(a); c = 0
    for _ in range(n):
        random.shuffle(p)
        if abs(st.median(p[:k]) - st.median(p[k:])) >= abs(obs) - 1e-12: c += 1
    return obs, (c + 1) / (n + 1)


def load(tag, s):
    if (tag, s) in CACHE: return CACHE[tag, s]
    R = []; d = os.path.join(RAW, tag, s)
    for i in range(1, 6):
        log = open(os.path.join(d, f'run_{i}.log'), errors='ignore').read()
        m = re.search(r'mean tracking time\s*:?\s*([0-9.e-]+)', log, re.I)
        traj = os.path.join(d, f'KeyFrameTrajectory_{i}.txt'); ft = os.path.join(d, f'frametimes_{i}.txt')
        fail = log.count('Fail to track local map')
        newmap = log.count('Creation of new map with id') - 1
        if not m or not os.path.exists(traj) or not os.path.exists(ft):
            R.append(dict(run=i, crash=True, fail=fail)); continue
        o = subprocess.run([sys.executable, '-I', '-X', 'utf8', ATE, os.path.join(GTD, GT[s], 'euroc_groundtruth.txt'), traj],
                           capture_output=True, text=True, encoding='utf-8').stdout
        a = re.search(r'([0-9.]+) m\s+n=', o)
        L = [l.split() for l in open(ft) if l.strip()]
        okr = 100 * sum(1 for x in L if x[4] == '2') / len(L)
        init = next((k for k, x in enumerate(L) if x[4] == '2'), len(L))
        R.append(dict(run=i, crash=False, mt=float(m.group(1)) * 1000, ate=float(a.group(1)) if a else None,
                      ok=okr, init=init, fail=fail, newmap=newmap))
    CACHE[tag, s] = R; return R


def ok(R, k): return [r[k] for r in R if not r['crash'] and r.get(k) is not None]
def rng(v, f='.4f'): return f'{st.median(v):{f}}（{min(v):{f}}–{max(v):{f}}，n={len(v)}）'


print('== 閃退（無軌跡或無 Mean tracking time）')
for tag in ['J1_1000', 'J1_0000'] + [f'J2_{c}_{m}' for c in ('C25', 'C35', 'S0') for m in ('ours1000', 'oursG10', 'orb3')]:
    for s in (list(GT)[:6] if tag.startswith('J1') else ['MH_03_medium', 'V2_02_medium']):
        for r in load(tag, s):
            if r['crash']: print(f'  {tag} {s} 第 {r["run"]} 次（log 內 Fail 計數 {r["fail"]}）')

J1S = list(GT)[:6]
print('\n== J1 逐序列（1000 vs 0000）')
for s in J1S:
    a, b = load('J1_1000', s), load('J1_0000', s)
    ea, eb = ok(a, 'ate'), ok(b, 'ate'); la, lb = ok(a, 'mt'), ok(b, 'mt'); d, p = perm(la, lb)
    print(f'-- {s}\n  ATE 1000 {rng(ea)}  0000 {rng(eb)}\n  延遲 ms 1000 {rng(la,".2f")}  0000 {rng(lb,".2f")}  差 {d:+.2f} p={p:.4f}'
          f'\n  初始化幀 1000 {rng(ok(a,"init"),".0f")}  0000 {rng(ok(b,"init"),".0f")}'
          f'\n  追蹤失敗 1000 {[r["fail"] for r in a]}  0000 {[r["fail"] for r in b]}')

print('\n== J1a：6 序列中至少 5 個，1000 的 ATE 中位數 ≤ 0000 的 5 次最大值')
n1a = 0
for s in J1S:
    ea, eb = ok(load('J1_1000', s), 'ate'), ok(load('J1_0000', s), 'ate'); good = st.median(ea) <= max(eb); n1a += good
    print(f'  {s}: 1000 中位數 {st.median(ea):.4f} vs 0000 最大 {max(eb):.4f} → {"✓" if good else "✗"}')
print(f'  J1a：{n1a}/6 → {"成立" if n1a >= 5 else "不成立"}')

print('\n== J1b：6 序列平均 1000 − 0000 ≤ 0.5 ms')
C = {s: (st.median(ok(load('J1_1000', s), 'mt')), st.median(ok(load('J1_0000', s), 'mt'))) for s in J1S}
A = st.mean(v[0] for v in C.values()); B = st.mean(v[1] for v in C.values()); cost = A - B
print(f'  1000 {A:.2f}  0000 {B:.2f}  成本 {cost:+.2f} ms（{100*cost/B:+.1f}%） → {"成立" if cost <= 0.5 else "不成立"}')
for s in J1S:
    lb = ok(load('J1_0000', s), 'mt')
    if len(lb) < 5:
        for lab, fill in (('補最小值（對 J1b 最不利）', min(lb)), ('補最大值', max(lb))):
            b2 = st.median(lb + [fill]); Bs = (B * 6 - C[s][1] + b2) / 6
            print(f'  敏感度 {s} {lab}：成本 {A-Bs:+.2f} ms → {"成立" if A-Bs <= 0.5 else "不成立"}')

print('\n== J1c：V203 追蹤失敗，1000 中位數 < 0000 中位數')
fa = [r['fail'] for r in load('J1_1000', 'V2_03_difficult')]; fb = [r['fail'] for r in load('J1_0000', 'V2_03_difficult') if not r['crash']]
print(f'  1000 {st.median(fa)} {fa}  0000（排除閃退）{st.median(fb)} {fb} → {"成立" if st.median(fa) < st.median(fb) else "不成立"}')
if len(fb) < 5:
    fb2 = fb + [min(fb)]  # 閃退當最差＝對 J1c 最不利：補最小
    print(f'  敏感度（閃退補 0000 最小值）0000 中位數 {st.median(fb2)} → {"成立" if st.median(fa) < st.median(fb2) else "不成立"}')
d, p = perm(fa, fb); print(f'  （附帶）置換差 {d:+.1f} p={p:.4f}')

print('\n== J1d：每個序列，1000 初始化幀中位數 − 0000 ≤ 100 幀')
n1d = 0
for s in J1S:
    ia, ib = ok(load('J1_1000', s), 'init'), ok(load('J1_0000', s), 'init'); d = st.median(ia) - st.median(ib); n1d += d <= 100
    print(f'  {s}: 1000 {sorted(ia)} 0000 {sorted(ib)} 差 {d:+.0f} → {"✓" if d <= 100 else "✗"}')
print(f'  J1d：{n1d}/6 → {"成立" if n1d == 6 else "不成立"}')

print('\n== J2 逐格（OK 幀比例%／ATE／延遲 ms／追蹤失敗／ORB3 新地圖數）')
M = ['ours1000', 'oursG10', 'orb3']
for C_ in ('C25', 'C35', 'S0'):
    for s in ('MH_03_medium', 'V2_02_medium'):
        print(f'-- {C_} {s}')
        for m in M:
            R = load(f'J2_{C_}_{m}', s)
            print(f'  {m:8s} OK {rng(ok(R,"ok"),".1f")}  ATE {rng(ok(R,"ate"))}  延遲 {rng(ok(R,"mt"),".2f")}  fail {[r["fail"] for r in R]}'
                  + (f'  新地圖 {[r["newmap"] for r in R if not r["crash"]]}' if m == 'orb3' else ''))

print('\n== J2a：C25，MH03 和 V202 兩序列 OK 幀中位數 1000 − G10 都 > 5 個百分點')
dd = {}
for s in ('MH_03_medium', 'V2_02_medium'):
    a, g = ok(load('J2_C25_ours1000', s), 'ok'), ok(load('J2_C25_oursG10', s), 'ok'); dd[s] = st.median(a) - st.median(g)
    print(f'  {s}: 1000 {st.median(a):.1f} G10 {st.median(g):.1f} 差 {dd[s]:+.1f}')
if all(v > 5 for v in dd.values()): print('  J2a → 成立')
else:
    print('  J2a → 不成立；判斷規則：' + (
        '兩序列都在 ±5 內' if all(abs(v) <= 5 for v in dd.values()) else
        '任一序列 G10 − 1000 > 5（全域低門檻較好）' if any(v < -5 for v in dd.values()) else
        '一序列成立、另一序列在 ±5 內'))
print('  置換檢定（C35＋C25 合併，每序列，Bonferroni 0.05/2 = 0.025；僅 J2a 成立時才用於主張）')
for s in ('MH_03_medium', 'V2_02_medium'):
    a = sum((ok(load(f'J2_{c}_ours1000', s), 'ok') for c in ('C35', 'C25')), []); g = sum((ok(load(f'J2_{c}_oursG10', s), 'ok') for c in ('C35', 'C25')), [])
    d, p = perm(a, g); print(f'  {s}: 差 {d:+.1f} p={p:.4f}')

print('\n== J2b：S0，MH03／V202 至少一個，G10 ATE 中位數 > 1000 的 ATE 最大值，或 G10 延遲中位數 − 1000 ≥ 0.5 ms')
hit = False
for s in ('MH_03_medium', 'V2_02_medium'):
    ea, eg = ok(load('J2_S0_ours1000', s), 'ate'), ok(load('J2_S0_oursG10', s), 'ate')
    la, lg = ok(load('J2_S0_ours1000', s), 'mt'), ok(load('J2_S0_oursG10', s), 'mt')
    c1 = st.median(eg) > max(ea); c2 = st.median(lg) - st.median(la) >= 0.5; hit |= c1 or c2
    print(f'  {s}: ATE G10 中位數 {st.median(eg):.4f} vs 1000 最大 {max(ea):.4f} → {"✓" if c1 else "✗"}；延遲差 {st.median(lg)-st.median(la):+.2f} ms → {"✓" if c2 else "✗"}')
print(f'  J2b → {"成立" if hit else "不成立"}')
