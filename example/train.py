from ultralytics import YOLO

def main():
    model = YOLO('yolov8n.pt')

    model.train(
        data='data.yaml',
        epochs=100,
        imgsz=640,
        batch=16,
        pretrained=True,
        name='finetune_run',
        exist_ok=True,
    )

if __name__ == "__main__":
    main()