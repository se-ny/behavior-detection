from ultralytics import YOLO

if __name__ == '__main__':
    # 학습
    model = YOLO('yolov8n.pt')

    model.train(
        data=r'C:\Users\samsungacademy608-1\Desktop\Dataset (2)\Dataset\object\data.yaml',
        epochs=50,
        imgsz=640,
        batch=16,
        name='weapon_detection',
        patience=10,
        device=0
    )

    # 평가
    metrics = model.val()
    print(f"\n🎉 mAP50: {metrics.box.map50:.4f}")
    print(f"   mAP50-95: {metrics.box.map:.4f}")