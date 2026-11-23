# [20261122] L2 區域性弱紋理退化（事先登記 20261121 事先登記 補實驗L1-L4 v2.md，參數照登記、不調）
#   退化區域＝影像左半（x < W/2），右半保持原圖；接縫 16 像素線性混合（x=W/2−8 … W/2+8）
#   RC：左半 I' = round(μ + 0.25(I − μ))，μ＝該張影像左半的平均灰階
#   RB：左半高斯模糊 σ＝3 像素（kernel 由 σ 決定）
# 原始資料不動；輸出 ~/slam_work/datasets/EuRoC_deg/<RC|RB>/<序列>/mav0/cam0/data
import os, glob, numpy as np, cv2
W_ = os.path.expanduser('~/slam_work'); SEQS = ['MH_03_medium', 'V2_02_medium']
for cond in ['RC', 'RB']:
    for s in SEQS:
        src = f'{W_}/datasets/EuRoC/{s}/mav0/cam0/data'; dst = f'{W_}/datasets/EuRoC_deg/{cond}/{s}/mav0/cam0/data'
        os.makedirs(dst, exist_ok=True); fs = sorted(glob.glob(f'{src}/*.png')); sd = []
        for f in fs:
            I = cv2.imread(f, cv2.IMREAD_UNCHANGED); h, w = I.shape[:2]; half = w // 2
            x = np.arange(w, dtype=np.float64); wt = np.clip((half + 8 - x) / 16.0, 0.0, 1.0)  # 左側 1 → 右側 0
            If = I.astype(np.float64)
            if cond == 'RC':
                mu = If[:, :half].mean(); D = mu + 0.25 * (If - mu)
            else:
                D = cv2.GaussianBlur(If, (0, 0), 3).astype(np.float64)
            O = np.clip(np.round(wt[None, :] * D + (1 - wt[None, :]) * If), 0, 255).astype(np.uint8)
            cv2.imwrite(f'{dst}/{os.path.basename(f)}', O)
            sd.append(O[:, :half - 8].std() / max(I[:, :half - 8].std(), 1e-9))
        print(cond, s, len(fs), '張；左半標準差比例中位數', round(float(np.median(sd)), 3), flush=True)
