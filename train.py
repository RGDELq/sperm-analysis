from ultralytics import YOLO

if __name__ == '__main__':

    # =========================
    # LOAD BASE MODEL
    # =========================
    model = YOLO("yolov8s.pt")

    # =========================
    # TRAIN
    # =========================
    model.train(

        data=r"C:\Users\AL-OFQ\sperm_analysis\dataset\dataset.yaml",

        # training
        epochs=50,
        patience=20,

        # tiny object detection
        imgsz=1024,

        # hardware
        batch=8,
        device=0,
        workers=1,

        # optimizer
        optimizer='AdamW',
        lr0=0.001,

        # augmentations
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,

        degrees=5,
        translate=0.1,
        scale=0.3,

        fliplr=0.5,

        mosaic=1.0,
        mixup=0.1,

        # save
        project=r"C:\Users\AL-OFQ\runs\detect",
        name="sperm_multiclass"
    )

    print("✅ TRAINING FINISHED!")