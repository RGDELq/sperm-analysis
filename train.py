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

        # use relative path to the dataset yaml (relative to this script's working directory)
        data=r"dataset/dataset.yaml",

        # training
        epochs=100,
        patience=30,

        # tiny object detection
        imgsz=1024,

        # hardware
        batch=8,
        device=0,
        workers=4,

        # optimizer
        optimizer='AdamW',
        lr0=0.001,

        # augmentations
        hsv_h=0.015,
        hsv_s=0.5,
        hsv_v=0.4,

        degrees=5,
        translate=0.1,
        scale=0.3,

        fliplr=0.5,

        mosaic=1.0,
        mixup=0,

        # save
        project=r"runs/detect",
        name="sperm_multiclass"
    )

    print("✅ TRAINING FINISHED!")