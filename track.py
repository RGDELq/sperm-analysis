from ultralytics import YOLO
from collections import defaultdict
from pathlib import Path
import cv2
import numpy as np
import csv

if __name__ == '__main__':

    model = YOLO(
        r"C:\Users\AL-OFQ\runs\detect\sperm_multiclass-2\weights\best.pt"
    )

    # =====================================================
    # ROOT DATASET PATH
    # =====================================================
    ROOT_PATH = Path(
        r"C:\Users\AL-OFQ\Downloads\spai_data\data\raw\Dataset"
    )

    # =====================================================
    # OUTPUT ROOT
    # =====================================================
    OUTPUT_ROOT = Path(
        r"C:\Users\AL-OFQ\sperm_analysis\tracked_output"
    )

    OUTPUT_ROOT.mkdir(exist_ok=True)

    # =====================================================
    # GET ALL VIDEOS
    # =====================================================
    videos = [
    v for v in ROOT_PATH.rglob("*.mp4")
    if "contrasto" not in v.stem.lower()
    and "contraso" not in v.stem.lower()
    ]
    print(f"✅ Found {len(videos)} videos")

    # =====================================================
    # LOOP OVER VIDEOS
    # =====================================================
    for video_index, VIDEO_PATH in enumerate(videos):

        print(f"\n🎥 [{video_index+1}/{len(videos)}]")
        print(f"📁 Processing: {VIDEO_PATH.name}")

        # =================================================
        # OUTPUT FOLDER FOR EACH VIDEO
        # =================================================
        video_output_dir = OUTPUT_ROOT / VIDEO_PATH.stem

        video_output_dir.mkdir(exist_ok=True)

        output_video = video_output_dir / "tracked_output.mp4"

        trajectory_csv = video_output_dir / "trajectories.csv"

        # =================================================
        # VIDEO INFO
        # =================================================
        cap = cv2.VideoCapture(str(VIDEO_PATH))

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30

        cap.release()

        # =================================================
        # VIDEO WRITER
        # =================================================
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        out = cv2.VideoWriter(
            str(output_video),
            fourcc,
            fps,
            (width, height)
        )

        # =================================================
        # CSV
        # =================================================
        csv_file = open(
            trajectory_csv,
            mode='w',
            newline=''
        )

        csv_writer = csv.writer(csv_file)

        csv_writer.writerow([
            "track_id",
            "frame",
            "x",
            "y",
            "confidence"
        ])

        # =================================================
        # TRACK HISTORY
        # =================================================
        track_history = defaultdict(list)

        frame_number = 0

        # =================================================
        # TRACKING
        # =================================================
        results = model.track(
            source=str(VIDEO_PATH),
            tracker="bytetrack.yaml",
            stream=True,
            persist=True,
            conf=0.10,
            iou=0.3,
            imgsz=1024,
            device=0,
            show=False,
            verbose=False
        )

        # =================================================
        # PROCESS FRAMES
        # =================================================
        for result in results:

            frame_number += 1

            frame = result.orig_img.copy()

            if result.boxes is not None and result.boxes.id is not None:

                boxes = result.boxes.xyxy.cpu().numpy()

                ids = result.boxes.id.int().cpu().tolist()

                confs = result.boxes.conf.cpu().numpy()

                classes = result.boxes.cls.cpu().numpy()

                for box, tid, conf, cls in zip(
                    boxes,
                    ids,
                    confs,
                    classes
                ):

                    if int(cls) != 0:
                        continue

                    x1, y1, x2, y2 = map(int, box)

                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2

                    track = track_history[tid]

                    track.append((cx, cy))

                    if len(track) > 15:
                        track.pop(0)

                    csv_writer.writerow([
                        tid,
                        frame_number,
                        cx,
                        cy,
                        round(float(conf), 4)
                    ])

                    # =====================================
                    # DRAW BOX
                    # =====================================
                    cv2.rectangle(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        (0, 255, 0),
                        2
                    )

                    # =====================================
                    # LABEL
                    # =====================================
                    cv2.putText(
                        frame,
                        f"id:{tid} {conf:.2f}",
                        (x1, max(y1 - 8, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2
                    )

                    # =====================================
                    # TRAJECTORY
                    # =====================================
                    if len(track) > 1:

                        pts = np.array(
                            track,
                            dtype=np.int32
                        )

                        cv2.polylines(
                            frame,
                            [pts],
                            False,
                            (255, 0, 255),
                            2
                        )

                    # =====================================
                    # CENTER POINT
                    # =====================================
                    cv2.circle(
                        frame,
                        (cx, cy),
                        4,
                        (0, 0, 255),
                        -1
                    )

            out.write(frame)

            print(
                f"Frame {frame_number} ✓",
                end='\r'
            )

        # =================================================
        # RELEASE
        # =================================================
        out.release()

        csv_file.close()

        print(f"\n Finished: {VIDEO_PATH.name}")

    print("\n ALL VIDEOS FINISHED!")