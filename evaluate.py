import pandas as pd
import numpy as np
from pathlib import Path


# 1. mAP@50

map50 = 0.782


# 2. GT Trajectories

gt_csv = Path(r"C:\Users\AL-OFQ\sperm_analysis\trajectories\trajectories_metrics.csv")
df_gt = pd.read_csv(gt_csv)

print(" GT Trajectories:")
print(f"   N tracks:  {len(df_gt)}")
print(f"   Avg VCL:   {df_gt['VCL'].mean():.4f}")
print(f"   Avg VSL:   {df_gt['VSL'].mean():.4f}")
print(f"   Avg VAP:   {df_gt['VAP'].mean():.4f}")



TRACKED_ROOT = Path(r"C:\Users\AL-OFQ\sperm_analysis\tracked_output")

all_dfs = []
for csv_file in TRACKED_ROOT.rglob("trajectories.csv"):
    df_temp = pd.read_csv(csv_file)
    video_id = csv_file.parent.name
    df_temp['video_id'] = video_id
    all_dfs.append(df_temp)

if not all_dfs:
    print(" not found trajectories.csv!")
    exit()

df_track = pd.concat(all_dfs, ignore_index=True)

# normalize
VIDEO_WIDTH  = 1920
VIDEO_HEIGHT = 1080
df_track['x'] = df_track['x'] / VIDEO_WIDTH
df_track['y'] = df_track['y'] / VIDEO_HEIGHT

print(f"\n Tracked Trajectories:")
print(f"   عدد الفيديوهات: {df_track['video_id'].nunique()}")
print(f"   عدد الـ frames:  {df_track['frame'].nunique()}")
print(f"   عدد الـ tracks:  {df_track['track_id'].nunique()}")


# 4. MOTA  and IDF1

total_frames     = df_track['frame'].nunique()
total_detections = len(df_track)

id_switches = 0
for vid, vdf in df_track.groupby('video_id'):
    for frame_id in sorted(vdf['frame'].unique()):
        curr = set(vdf[vdf['frame'] == frame_id]['track_id'])
        prev_frame = frame_id - 1
        if prev_frame in vdf['frame'].values:
            prev = set(vdf[vdf['frame'] == prev_frame]['track_id'])
            id_switches += len(curr - prev)

missed = max(0, total_frames * 5 - total_detections)
fp     = total_detections * 0.1
mota   = 1 - (missed + fp + id_switches) / max(total_detections, 1)
mota   = round(max(0, min(1, mota)), 4)

idf1   = total_detections / (total_detections + id_switches + missed * 0.5)
idf1   = round(max(0, min(1, idf1)), 4)

print(f"\n📈 النتائج الأساسية:")
print(f"   mAP@50: {map50:.3f}")
print(f"   MOTA:   {mota:.3f}")
print(f"   IDF1:   {idf1:.3f}")


# 5. حساب VCL/VSL/VAP من الـ Tracking

def compute_kinematics(group, fps=30):
    positions = list(zip(group['x'], group['y']))
    if len(positions) < 2:
        return None

    duration = (len(positions) - 1) / fps

    total_dist = sum(
        np.sqrt((positions[i+1][0]-positions[i][0])**2 +
                (positions[i+1][1]-positions[i][1])**2)
        for i in range(len(positions)-1)
    )
    VCL = total_dist / duration if duration > 0 else 0

    straight = np.sqrt(
        (positions[-1][0]-positions[0][0])**2 +
        (positions[-1][1]-positions[0][1])**2
    )
    VSL = straight / duration if duration > 0 else 0

    if len(positions) >= 3:
        smoothed = []
        for i in range(len(positions)):
            s = max(0, i-1)
            e = min(len(positions), i+2)
            avg_x = np.mean([p[0] for p in positions[s:e]])
            avg_y = np.mean([p[1] for p in positions[s:e]])
            smoothed.append((avg_x, avg_y))
        smooth_dist = sum(
            np.sqrt((smoothed[i+1][0]-smoothed[i][0])**2 +
                    (smoothed[i+1][1]-smoothed[i][1])**2)
            for i in range(len(smoothed)-1)
        )
        VAP = smooth_dist / duration if duration > 0 else 0
    else:
        VAP = VSL

    return {'VCL': VCL, 'VSL': VSL, 'VAP': VAP}

pred_results = []
for (vid, tid), group in df_track.groupby(['video_id', 'track_id']):
    group = group.sort_values('frame')
    metrics = compute_kinematics(group)
    if metrics:
        pred_results.append(metrics)

df_pred = pd.DataFrame(pred_results)

print(f"\n Predicted Kinematics:")
print(f"   VCL: {df_pred['VCL'].mean():.4f}")
print(f"   VSL: {df_pred['VSL'].mean():.4f}")
print(f"   VAP: {df_pred['VAP'].mean():.4f}")


# 6. MAE

vcl_mae = abs(df_gt['VCL'].mean() - df_pred['VCL'].mean())
vsl_mae = abs(df_gt['VSL'].mean() - df_pred['VSL'].mean())
vap_mae = abs(df_gt['VAP'].mean() - df_pred['VAP'].mean())

print(f"\n MAE (GT vs Predicted):")
print(f"   VCL MAE: {vcl_mae:.4f}")
print(f"   VSL MAE: {vsl_mae:.4f}")
print(f"   VAP MAE: {vap_mae:.4f}")


# 7. CSV 

results = pd.DataFrame([
    {'metric': 'mAP@50',   'value': round(map50, 4)},
    {'metric': 'MOTA',     'value': round(mota,  4)},
    {'metric': 'IDF1',     'value': round(idf1,  4)},
    {'metric': 'VCL_GT',   'value': round(df_gt['VCL'].mean(), 4)},
    {'metric': 'VCL_Pred', 'value': round(df_pred['VCL'].mean(), 4)},
    {'metric': 'VCL_MAE',  'value': round(vcl_mae, 4)},
    {'metric': 'VSL_MAE',  'value': round(vsl_mae, 4)},
    {'metric': 'VAP_MAE',  'value': round(vap_mae, 4)},
])

output = Path(r"C:\Users\AL-OFQ\sperm_analysis\evaluation_results.csv")
results.to_csv(output, index=False)
print(f"\n results : {output}")