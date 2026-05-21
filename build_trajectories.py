import os
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import linear_sum_assignment

PROCESSED_PATH = Path(r"C:\Users\AL-OFQ\Downloads\spai_data\data\processed")
OUTPUT_PATH = Path(r"C:\Users\AL-OFQ\sperm_analysis\trajectories")
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

def read_bbox_file(filepath):
    boxes = []
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                cls, x, y, w, h = map(float, parts)
                boxes.append({
                    'class': int(cls),
                    'x': x, 'y': y,
                    'w': w, 'h': h
                })
    return boxes


def compute_iou(b1, b2):
    x1_min = b1['x'] - b1['w']/2
    x1_max = b1['x'] + b1['w']/2
    y1_min = b1['y'] - b1['h']/2
    y1_max = b1['y'] + b1['h']/2

    x2_min = b2['x'] - b2['w']/2
    x2_max = b2['x'] + b2['w']/2
    y2_min = b2['y'] - b2['h']/2
    y2_max = b2['y'] + b2['h']/2

    inter_x = max(0, min(x1_max, x2_max) - max(x1_min, x2_min))
    inter_y = max(0, min(y1_max, y2_max) - max(y1_min, y2_min))
    inter = inter_x * inter_y

    area1 = b1['w'] * b1['h']
    area2 = b2['w'] * b2['h']
    union = area1 + area2 - inter

    return inter / union if union > 0 else 0

# ============================================
# Frame matching
# ============================================
def match_boxes(prev_boxes, curr_boxes):
    if not prev_boxes or not curr_boxes:
        return []
    
    # نبني matrix of IoU
    iou_matrix = np.zeros((len(prev_boxes), len(curr_boxes)))
    for i, b1 in enumerate(prev_boxes):
        for j, b2 in enumerate(curr_boxes):
            iou_matrix[i][j] = compute_iou(b1, b2)
    
    # Hungarian algorithm
    row_ind, col_ind = linear_sum_assignment(-iou_matrix)
    
    matches = []
    for r, c in zip(row_ind, col_ind):
        if iou_matrix[r][c] > 0.1:  # threshold
            matches.append((r, c))
    
    return matches

def build_trajectories_for_video(frame_files):
    # نرتب الframes بالترتيب الرقمي
    frame_files = sorted(frame_files, key=lambda x: int(
        ''.join(filter(str.isdigit, x.stem.split('chiaro')[-1])) or '0'
    ))
    
    tracks = {}      # track_id -> list of positions
    next_id = 0
    prev_boxes = []
    prev_ids = []

    for frame_idx, ffile in enumerate(frame_files):
        curr_boxes = read_bbox_file(ffile)
        
        if frame_idx == 0:
            for box in curr_boxes:
                tracks[next_id] = [{
                    'frame': frame_idx,
                    'x': box['x'],
                    'y': box['y'],
                    'w': box['w'],
                    'h': box['h']
                }]
                prev_ids.append(next_id)
                next_id += 1
        else:
            matches = match_boxes(prev_boxes, curr_boxes)
            matched_curr = set()
            new_prev_ids = []

            for (r, c) in matches:
                tid = prev_ids[r]
                tracks[tid].append({
                    'frame': frame_idx,
                    'x': curr_boxes[c]['x'],
                    'y': curr_boxes[c]['y'],
                    'w': curr_boxes[c]['w'],
                    'h': curr_boxes[c]['h']
                })
                matched_curr.add(c)
                new_prev_ids.append((c, tid))

            # خلايا جديدة ما عندها match
            for c, box in enumerate(curr_boxes):
                if c not in matched_curr:
                    tracks[next_id] = [{
                        'frame': frame_idx,
                        'x': box['x'],
                        'y': box['y'],
                        'w': box['w'],
                        'h': box['h']
                    }]
                    new_prev_ids.append((c, next_id))
                    next_id += 1

            new_prev_ids.sort(key=lambda x: x[0])
            prev_ids = [tid for _, tid in new_prev_ids]

        prev_boxes = curr_boxes

    return tracks

# ============================================
# حساب VCL / VSL / VAP
# ============================================
def compute_kinematics(track_points, fps=30):
    if len(track_points) < 2:
        return None

    positions = [(p['x'], p['y']) for p in track_points]

    # VCL - مجموع كل المسافات بين نقطتين متتاليتين
    total_dist = sum(
        np.sqrt((positions[i+1][0]-positions[i][0])**2 +
                (positions[i+1][1]-positions[i][1])**2)
        for i in range(len(positions)-1)
    )
    duration = (len(positions)-1) / fps
    VCL = total_dist / duration if duration > 0 else 0

    # VSL - المسافة المستقيمة من أول نقطة لآخر نقطة
    straight_dist = np.sqrt(
        (positions[-1][0]-positions[0][0])**2 +
        (positions[-1][1]-positions[0][1])**2
    )
    VSL = straight_dist / duration if duration > 0 else 0

    # VAP - متوسط المسار المنعّم (moving average)
    if len(positions) >= 3:
        smoothed = []
        for i in range(len(positions)):
            start = max(0, i-1)
            end = min(len(positions), i+2)
            avg_x = np.mean([p[0] for p in positions[start:end]])
            avg_y = np.mean([p[1] for p in positions[start:end]])
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

# ============================================
# الحلقة الرئيسية - نمشي على كل الفيديوهات
# ============================================
all_results = []

# نجمع كل الـ bbox files وننظمهم حسب الفيديو
video_frames = {}

for bbox_file in PROCESSED_PATH.rglob("*chiaro*_bbox.txt"):
    # اسم الفيديو = اسم المريض + رقم الفيديو
    parts = bbox_file.stem.split('chiaro')
    video_name = parts[0].strip() + "_" + bbox_file.parent.name
    
    if video_name not in video_frames:
        video_frames[video_name] = []
    video_frames[video_name].append(bbox_file)

print(f"✅ n of videos {len(video_frames)} ")

# نبني trajectories لكل فيديو
for video_name, frame_files in video_frames.items():
    print(f"🎬 processing: {video_name} ({len(frame_files)} frames)")
    
    tracks = build_trajectories_for_video(frame_files)
    
    for track_id, points in tracks.items():
        if len(points) < 2:
            continue
            
        metrics = compute_kinematics(points, fps=30)
        if metrics:
            all_results.append({
                'video_id': video_name,
                'track_id': track_id,
                'num_frames': len(points),
                'VCL': round(metrics['VCL'], 4),
                'VSL': round(metrics['VSL'], 4),
                'VAP': round(metrics['VAP'], 4),
            })

# ============================================
# تصدير CSV
# ============================================
df = pd.DataFrame(all_results)
csv_path = OUTPUT_PATH / "trajectories_metrics.csv"
df.to_csv(csv_path, index=False)

print(f"\n Results  {csv_path}")
print(f"📊N of tracks: {len(df)}")
print(f"\n matrices avrage")
print(f"   VCL: {df['VCL'].mean():.4f}")
print(f"   VSL: {df['VSL'].mean():.4f}")
print(f"   VAP: {df['VAP'].mean():.4f}")