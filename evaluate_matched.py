import pandas as pd
import numpy as np
from pathlib import Path
from scipy.optimize import linear_sum_assignment


# 1.reading the GT

gt_csv = Path(r"C:\Users\AL-OFQ\sperm_analysis\trajectories\trajectories_metrics.csv")
df_gt = pd.read_csv(gt_csv)

print(f"📊 GT tracks: {len(df_gt)}")


# 2.reading all Predicted CSVs

TRACKED_ROOT = Path(r"C:\Users\AL-OFQ\sperm_analysis\tracked_output")

all_dfs = []
for csv_file in TRACKED_ROOT.rglob("trajectories.csv"):

    video_id = csv_file.parent.name

    # working on CHIARO
    if "CONTRASTO" in video_id.upper():
        continue

    df_temp = pd.read_csv(csv_file)
    df_temp['video_id'] = video_id

    all_dfs.append(df_temp)

df_track = pd.concat(all_dfs, ignore_index=True)

# normalize
VIDEO_WIDTH  = 1920
VIDEO_HEIGHT = 1080
df_track['x'] = df_track['x'] / VIDEO_WIDTH
df_track['y'] = df_track['y'] / VIDEO_HEIGHT


# 3. computing  VCL/VSL/VAP from Predicted

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

# comput for every metrices predicted track
pred_results = []
for (vid, tid), group in df_track.groupby(['video_id', 'track_id']):
    group = group.sort_values('frame')
    metrics = compute_kinematics(group)
    if metrics:
        pred_results.append({
            'video_id': vid,
            'track_id': tid,
            'VCL': metrics['VCL'],
            'VSL': metrics['VSL'],
            'VAP': metrics['VAP']
        })

df_pred = pd.DataFrame(pred_results)
print(f"📊 Predicted tracks: {len(df_pred)}")


# 4. Hungarian Matching per video

print("\n Hungarian Matching...")

matched_errors = []

# all videos
import re
import re

for video_id in df_pred['video_id'].unique():

    # predicted tracks 
    pred_vid = df_pred[df_pred['video_id'] == video_id]

    # the base id exstraction 
    match = re.search(
        r'[A-Z]{2}\s?\d+',
        str(video_id)
    )

    if match:
        pred_clean = (
            match.group(0)
            .replace(' ', '')
            .lower()
        )
    else:
        print(f" Could not parse: {video_id}")
        continue

    # clean GT
    gt_temp = df_gt.copy()

    gt_temp['clean_name'] = gt_temp['video_id'].apply(
        lambda x: (
            re.search(
                r'[A-Z]{2}\s?\d+',
                str(x)
            ).group(0)
            .replace(' ', '')
            .lower()
            if re.search(
                r'[A-Z]{2}\s?\d+',
                str(x)
            )
            else ''
        )
    )

    # matching
    gt_vid = gt_temp[
        gt_temp['clean_name'] == pred_clean
    ]

    print(f"\n🔍 {video_id}")
    print(f"   Parsed ID: {pred_clean}")
    print(f"   Matches found: {len(gt_vid)}")


    if len(gt_vid) == 0 or len(pred_vid) == 0:
        print(f"❌ No match for: {video_id}")
        continue

    #  cost matrix based on VCL difference
    gt_vcl   = gt_vid['VCL'].values
    pred_vcl = pred_vid['VCL'].values

    cost_matrix = np.abs(
        gt_vcl[:, None] - pred_vcl[None, :]
    )

    # Hungarian Algorithm
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    for r, c in zip(row_ind, col_ind):

        matched_errors.append({

            'video_id': video_id,

            'gt_VCL': gt_vcl[r],
            'pred_VCL': pred_vcl[c],
            'VCL_error': abs(
                gt_vcl[r] - pred_vcl[c]
            ),

            'gt_VSL': gt_vid['VSL'].values[r],
            'pred_VSL': pred_vid['VSL'].values[c],
            'VSL_error': abs(
                gt_vid['VSL'].values[r] -
                pred_vid['VSL'].values[c]
            ),

            'gt_VAP': gt_vid['VAP'].values[r],
            'pred_VAP': pred_vid['VAP'].values[c],
            'VAP_error': abs(
                gt_vid['VAP'].values[r] -
                pred_vid['VAP'].values[c]
            ),
        })
df_matched = pd.DataFrame(matched_errors)


# 5. resulta

if len(df_matched) > 0:
    vcl_mae = df_matched['VCL_error'].mean()
    vsl_mae = df_matched['VSL_error'].mean()
    vap_mae = df_matched['VAP_error'].mean()

    print(f"\n Per-Object MAE (Hungarian Matched):")
    print(f"   VCL MAE: {vcl_mae:.4f}")
    print(f"   VSL MAE: {vsl_mae:.4f}")
    print(f"   VAP MAE: {vap_mae:.4f}")
    print(f"   Matched pairs: {len(df_matched)}")

    
    output = Path(r"C:\Users\AL-OFQ\sperm_analysis\evaluation_matched.csv")
    df_matched.to_csv(output, index=False)
    print(f"\n saved: {output}")
else:
    print("checking if of GT")
    print(f"GT columns: {df_gt.columns.tolist()}")
    print(f"GT sample:\n{df_gt.head()}")