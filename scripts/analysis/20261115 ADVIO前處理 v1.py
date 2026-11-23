# ADVIO 前處理（依先前實作者 Tool.zip/ADVIO/Video2Image.py 的做法，另外縮成 360x640）
# 用法：python3 convert_advio.py <ADVIO 資料夾> <序列名...>   例：... ~/slam_work/datasets/ADVIO advio-01 advio-02
# 產出（每個序列）：
#   iphone/img360/<n>.png        影片第 n 幀（1 起算，同 Video2Image.py 的命名），縮成 360x640（cv2.resize 預設雙線性，同 mono_advio 的 newWidth 縮放）
#   iphone/frames_3500.csv       frames.csv 的前 3500 行（先前實作者 ADVIO 實驗只跑前 3500 幀）
#   ground-truth/groundtruth_tum.txt   pose.csv（time,x,y,z,四元數）→ TUM 格式（ATE 只用位置）
import sys, os, cv2, csv

root, seqs = sys.argv[1], sys.argv[2:]
for seq in seqs:
    d = os.path.join(root, seq)
    out = os.path.join(d, 'iphone', 'img360')
    os.makedirs(out, exist_ok=True)
    cap = cv2.VideoCapture(os.path.join(d, 'iphone', 'frames.mov'))
    n = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        n += 1
        if fr.shape[1] != 720 or fr.shape[0] != 1280:
            raise SystemExit(f'{seq}: 第 {n} 幀尺寸 {fr.shape} 不是 720x1280')
        cv2.imwrite(os.path.join(out, f'{n}.png'), cv2.resize(fr, (360, 640)))
    cap.release()
    rows = [r for r in csv.reader(open(os.path.join(d, 'iphone', 'frames.csv'))) if r]
    with open(os.path.join(d, 'iphone', 'frames_3500.csv'), 'w', newline='') as f:
        csv.writer(f).writerows(rows[:3500])
    with open(os.path.join(d, 'ground-truth', 'groundtruth_tum.txt'), 'w') as f:
        f.write('# timestamp tx ty tz q1 q2 q3 q4（由 ADVIO pose.csv 轉換；四元數順序照原檔，ATE 只用位置）\n')
        for r in csv.reader(open(os.path.join(d, 'ground-truth', 'pose.csv'))):
            if r:
                f.write(' '.join(x.strip() for x in r[:8]) + '\n')
    last_idx = int(rows[-1][1])
    print(f'{seq}: 影片 {n} 幀；frames.csv {len(rows)} 行（最後索引 {last_idx}）；'
          + ('OK' if last_idx <= n else f'⚠️ csv 索引超過影片幀數'))
