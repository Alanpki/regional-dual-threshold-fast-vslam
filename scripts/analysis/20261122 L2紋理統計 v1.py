# [20261122] L2 區域退化影像的紋理統計（事先登記 20261121 事先登記 補實驗L1-L4 v2.md：每格 FAST-20 角點可用率、低門檻回收格子比例）
# 方法同 20261116 紋理統計 v1.py：每 10 幀取 1 幀、灰階、OpenCV FAST（9-16、NMS）門檻 20／10、32×32 格
#   左半＝x < W/2−8（退化區）、右半＝x ≥ W/2+8（原圖）；接縫 16 像素不計。格子以各半邊的左上角起算。
#   FAST-20 可用率＝門檻 20 至少有一個角點的格子比例；低門檻回收＝門檻 20 沒有、門檻 10 有角點的格子比例（占全部格子）
import os, glob, statistics as st, numpy as np, cv2
W = os.path.expanduser('~/slam_work/datasets')
f20 = cv2.FastFeatureDetector_create(20, True); f10 = cv2.FastFeatureDetector_create(10, True)
def cells(kps, h, w, c=32):
    g = np.zeros(((h + c - 1) // c, (w + c - 1) // c), bool)
    for k in kps: g[int(k.pt[1]) // c, int(k.pt[0]) // c] = True
    return g
def stat(im):
    h, w = im.shape; c20, c10 = cells(f20.detect(im), h, w), cells(f10.detect(im), h, w)
    return c20.mean() * 100, (~c20 & c10).mean() * 100, (~c20 & ~c10).mean() * 100
print('| 序列 | 條件 | 區域 | FAST-20 可用格 % | 低門檻回收 % | 兩門檻都空 % |'); print('|---|---|---|---|---|---|')
for s in ('MH_03_medium', 'V2_02_medium'):
    for cond, root in (('原圖', f'{W}/EuRoC'), ('RC', f'{W}/EuRoC_deg/RC'), ('RB', f'{W}/EuRoC_deg/RB')):
        fs = sorted(glob.glob(f'{root}/{s}/mav0/cam0/data/*.png'))[::10]; R = {'左半': [], '右半': []}
        for p in fs:
            im = cv2.imread(p, cv2.IMREAD_GRAYSCALE); w = im.shape[1]; hf = w // 2
            R['左半'].append(stat(np.ascontiguousarray(im[:, :hf - 8]))); R['右半'].append(stat(np.ascontiguousarray(im[:, hf + 8:])))
        for k in ('左半', '右半'):
            a = np.median(np.array(R[k]), axis=0); print(f'| {s} | {cond} | {k}（{len(fs)} 幀） | {a[0]:.1f} | {a[1]:.1f} | {a[2]:.1f} |')
