# [20261117] A 退化掃描：對比降低 I' = round(μ + α(I − μ))，μ＝每張影像平均灰階；原始資料不動，寫到 EuRoC_deg/<條件>/<序列>/mav0/cam0/data
#            跳幀：時間戳每 k 行取 1 行，寫到 EuRoC_ts/<條件>/（整個 EuRoC_TimeStamps 資料夾的副本）
import sys, os, glob, shutil, numpy as np, cv2
W = os.path.expanduser('~/slam_work')
SEQS = ['MH_03_medium', 'V2_02_medium']
for a in [0.5, 0.35, 0.25]:
    cond = f'C{int(round(a*100))}'
    for s in SEQS:
        src = f'{W}/datasets/EuRoC/{s}/mav0/cam0/data'; dst = f'{W}/datasets/EuRoC_deg/{cond}/{s}/mav0/cam0/data'
        os.makedirs(dst, exist_ok=True)
        fs = sorted(glob.glob(f'{src}/*.png')); sd = []
        for f in fs:
            I = cv2.imread(f, cv2.IMREAD_UNCHANGED).astype(np.float64); mu = I.mean()
            J = np.clip(np.round(mu + a * (I - mu)), 0, 255).astype(np.uint8)
            cv2.imwrite(f'{dst}/{os.path.basename(f)}', J); sd.append(J.std() / max(I.std(), 1e-9))
        print(cond, s, len(fs), '張；標準差比例中位數', round(float(np.median(sd)), 3), flush=True)
TS = f'{W}/code_v2/Examples/Monocular/EuRoC_TimeStamps'
for k in [2, 3, 4]:
    d = f'{W}/datasets/EuRoC_ts/K{k}'; os.makedirs(d, exist_ok=True)
    for f in glob.glob(f'{TS}/*.txt'):
        L = [l for l in open(f) if l.strip()]
        open(f'{d}/{os.path.basename(f)}', 'w').writelines(L[::k])
    print(f'K{k}', 'MH03', sum(1 for _ in open(f'{d}/MH03.txt')), '幀；V202', sum(1 for _ in open(f'{d}/V202.txt')), '幀')
