# [20261116] R1-2 紋理豐富度統計（與方法無關，只看影像）＋各方法每幀特徵點數（來自實驗的 frametimes）
# 影像：每 10 幀取 1 幀、灰階、原解析度（EuRoC 752x480、TUM 640x480、ADVIO 360x640）
#   grad   ：Sobel 梯度強度平均（0–255 尺度）
#   F20/F10：OpenCV FAST（9-16、非極大值抑制）門檻 20／10 的角點數，每 1 萬像素
#   空格20 ：32x32 網格中門檻 20 找不到任何角點的格子比例
#   補回   ：門檻 20 沒有、門檻 10 有角點的格子比例（雙門檻 20/10 能補回的格子）
import os, glob, statistics as st, numpy as np, cv2
W = os.path.expanduser('~/slam_work')
SEQ = [('EuRoC', s, f'{W}/datasets/EuRoC/{s}/mav0/cam0/data/*.png') for s in
       ['MH_01_easy', 'MH_02_easy', 'MH_03_medium', 'MH_04_difficult', 'MH_05_difficult', 'V2_03_difficult']]
SEQ += [('TUM', s.replace('rgbd_dataset_freiburg3_', ''), f'{W}/datasets/TUM/{s}/rgb/*.png') for s in
        sorted(os.path.basename(p) for p in glob.glob(f'{W}/datasets/TUM/rgbd_dataset_*') if os.path.isdir(p))]
SEQ += [('ADVIO', s, f'{W}/datasets/ADVIO/{s}/iphone/img360/*.png') for s in ['advio-01', 'advio-02', 'advio-03', 'advio-04']]
f20 = cv2.FastFeatureDetector_create(20, True); f10 = cv2.FastFeatureDetector_create(10, True)

def cells(kps, h, w, c=32):
    g = np.zeros(((h + c - 1) // c, (w + c - 1) // c), bool)
    for k in kps:
        g[int(k.pt[1]) // c, int(k.pt[0]) // c] = True
    return g

def feat(tag, seq):
    out = {}
    for name, pat in [('Jetson-SLAM', {'EuRoC': 'M1_baseline', 'TUM': 'TUM_baseline', 'ADVIO': 'ADVIO_B_baseline'}),
                      ('本方法', {'EuRoC': 'M1_ours', 'TUM': 'TUM_ours', 'ADVIO': 'ADVIO_B_ours'}),
                      ('ORB-SLAM3', {'EuRoC': 'M1_orb3', 'TUM': 'TUM_orb3', 'ADVIO': 'ADVIO_B_orb3'}),
                      ('單門檻20/20', {'V2': 'ABL_0000'}), ('雙門檻20/10', {'V2': 'ABL_1000'})]:
        key = 'V2' if seq.startswith(('V2', 'MH_04')) else tag
        if key not in pat or (key == 'V2') != (name.startswith(('單', '雙'))):
            continue
        fs = glob.glob(f'{W}/output/{pat[key]}/*{seq}*/frametimes_*.txt')
        v = [int(l.split()[2]) for f in fs for l in open(f) if l.strip() and len(l.split()) >= 5]
        if v:
            out[name] = st.median(v)
    return out

print('| 資料集 | 序列 | 取樣幀 | 梯度 | FAST20／萬px | FAST10／萬px | 空格（門檻20） | 雙門檻補回 | 每幀特徵點（中位數） |')
print('|---|---|---|---|---|---|---|---|---|')
for tag, seq, pat in SEQ:
    files = sorted(glob.glob(pat))[::10]
    G, A, B, E, R = [], [], [], [], []
    for p in files:
        im = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        h, w = im.shape
        gx = cv2.Sobel(im, cv2.CV_32F, 1, 0); gy = cv2.Sobel(im, cv2.CV_32F, 0, 1)
        G.append(float(np.mean(np.sqrt(gx * gx + gy * gy))) / 4)  # Sobel 3x3 最大增益 4 → 0–255 尺度
        k20, k10 = f20.detect(im), f10.detect(im)
        A.append(len(k20) / (h * w) * 1e4); B.append(len(k10) / (h * w) * 1e4)
        c20, c10 = cells(k20, h, w), cells(k10, h, w)
        E.append((~c20).mean() * 100); R.append((~c20 & c10).mean() * 100)
    fe = feat(tag, seq)
    print(f'| {tag} | {seq} | {len(files)} | {st.median(G):.1f} | {st.median(A):.1f} | {st.median(B):.1f} | {st.median(E):.0f}% | {st.median(R):.0f}% | '
          + '、'.join(f'{k} {v:.0f}' for k, v in fe.items()) + ' |', flush=True)
