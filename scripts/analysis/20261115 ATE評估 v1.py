"""ATE RMSE 評估（單目 SLAM 用 Sim(3) Umeyama 對齊）
用法：python "20261115 ATE評估 v1.py" <groundtruth.txt> <trajectory.txt> [<trajectory2.txt> ...]
格式：TUM（timestamp tx ty tz qx qy qz qw），時間單位秒
輸出：每條軌跡的 ATE RMSE（m）、對到的位姿數、時間覆蓋率（估計軌跡時間跨度 / GT 時間跨度）
"""
import sys
import numpy as np


def load_tum(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            v = line.replace(',', ' ').split()
            if len(v) < 4:
                continue
            rows.append([float(x) for x in v[:4]])
    a = np.array(rows)
    if a[0, 0] > 1e12:  # 奈秒 → 秒
        a[:, 0] *= 1e-9
    return a


def associate(gt, est, max_dt=0.02):
    idx = np.searchsorted(gt[:, 0], est[:, 0])
    idx = np.clip(idx, 1, len(gt) - 1)
    left, right = gt[idx - 1, 0], gt[idx, 0]
    idx = np.where(np.abs(est[:, 0] - left) < np.abs(est[:, 0] - right), idx - 1, idx)
    ok = np.abs(gt[idx, 0] - est[:, 0]) < max_dt
    return gt[idx[ok], 1:4], est[ok, 1:4]


def umeyama_sim3(src, dst):
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    cov = xd.T @ xs / len(src)
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    var_s = (xs ** 2).sum() / len(src)
    s = np.trace(np.diag(D) @ S) / var_s
    t = mu_d - s * R @ mu_s
    return s, R, t


def ate(gt, est):
    g, e = associate(gt, est)
    s, R, t = umeyama_sim3(e, g)
    aligned = (s * (R @ e.T)).T + t
    err = np.linalg.norm(aligned - g, axis=1)
    cover = (est[-1, 0] - est[0, 0]) / (gt[-1, 0] - gt[0, 0])
    return np.sqrt((err ** 2).mean()), len(g), cover


if __name__ == '__main__':
    gt = load_tum(sys.argv[1])
    for p in sys.argv[2:]:
        r, n, c = ate(gt, load_tum(p))
        print(f'{r:.4f} m  n={n:4d}  cover={c*100:5.1f}%  {p}')
