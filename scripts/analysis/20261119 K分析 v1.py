# [20261119] K 分析：依 20261119 事先登記 C25 V202補跑 v1.md（主判定只用新增 10 次；J 的 5 次只作描述）
import os, re, random, statistics as st, subprocess, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '4 第二次投稿'))
K = os.path.join(ROOT, '2 實驗結果', '20261119 K原始輸出 v1'); J = os.path.join(ROOT, '2 實驗結果', '20261119 J原始輸出 v1')
GT = os.path.join(ROOT, '2 實驗結果', '20261118 EuRoC groundtruth v1', 'V_202', 'euroc_groundtruth.txt'); ATE = os.path.join(ROOT, '1 實驗腳本', '20261115 ATE評估 v1.py')
S = 'V2_02_medium'
def perm(a, b, n=20000, seed=1):
    random.seed(seed); obs = st.median(a) - st.median(b); p = a + b; k = len(a); c = 0
    for _ in range(n):
        random.shuffle(p)
        if abs(st.median(p[:k]) - st.median(p[k:])) >= abs(obs) - 1e-12: c += 1
    return obs, (c + 1) / (n + 1)
def load(base, tag, runs):
    R = []; d = os.path.join(base, tag, S)
    for i in runs:
        log = open(f'{d}/run_{i}.log', errors='ignore').read(); m = re.search(r'mean tracking time\s*:?\s*([0-9.e-]+)', log, re.I)
        tj = f'{d}/KeyFrameTrajectory_{i}.txt'; ft = f'{d}/frametimes_{i}.txt'
        if not m or not os.path.exists(tj) or not os.path.exists(ft): R.append(dict(crash=True)); continue
        o = subprocess.run([sys.executable, '-I', '-X', 'utf8', ATE, GT, tj], capture_output=True, text=True, encoding='utf-8').stdout
        a = re.search(r'([0-9.]+) m\s+n=', o); L = [l.split() for l in open(ft) if l.strip()]
        R.append(dict(crash=False, ok=100*sum(1 for x in L if x[4]=='2')/len(L), ate=float(a.group(1)) if a else None, mt=float(m.group(1))*1000, fail=log.count('Fail to track local map')))
    return R
def g(R, k): return [r[k] for r in R if not r['crash'] and r.get(k) is not None]
def rg(v, f='.1f'): return f'{st.median(v):{f}}（{min(v):{f}}–{max(v):{f}}，n={len(v)}）'
A = load(K, 'K_C25_ours1000', range(6, 16)); B = load(K, 'K_C25_oursG10', range(6, 16))
print('== 閃退：', sum(r['crash'] for r in A), '(1000)', sum(r['crash'] for r in B), '(G10)')
print('== 新增 10 次（主判定）')
print(f'  OK 幀% 1000 {rg(g(A,"ok"))}  G10 {rg(g(B,"ok"))}')
print(f'  追蹤失敗 1000 {[r["fail"] for r in A]}  G10 {[r["fail"] for r in B]}')
print(f'  ATE 1000 {rg(g(A,"ate"),".4f")}  G10 {rg(g(B,"ate"),".4f")}')
print(f'  延遲 ms 1000 {rg(g(A,"mt"),".2f")}  G10 {rg(g(B,"mt"),".2f")}')
d, p = perm(g(B, 'ok'), g(A, 'ok')); print(f'  G10 − 1000 OK 幀中位數差 {d:+.1f} 個百分點，置換 p={p:.4f}')
if d > 5 and p < 0.05: v = 'K1 成立（全域低門檻較好）'
elif abs(d) <= 5: v = 'K2 成立（相近）'
elif d < -5 and p < 0.05: v = 'K3 成立（雙門檻較好）→ 先停下來討論'
else: v = 'K4（差 > 5 但 p ≥ 0.05，不確定）'
print('  判定 →', v)
JA = load(J, 'J2_C25_ours1000', range(1, 6)); JB = load(J, 'J2_C25_oursG10', range(1, 6))
a15, b15 = g(JA, 'ok') + g(A, 'ok'), g(JB, 'ok') + g(B, 'ok'); d2, p2 = perm(b15, a15)
print(f'== 描述：併入 J 的 5 次（各 15 次）OK 幀 1000 {rg(a15)} G10 {rg(b15)} 差 {d2:+.1f} p={p2:.4f}')
