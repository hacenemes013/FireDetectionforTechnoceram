from ultralytics import YOLO
import os
import yaml

os.makedirs('dataset/images/train', exist_ok=True)
os.makedirs('dataset/images/val', exist_ok=True)
os.makedirs('dataset/labels/train', exist_ok=True)
os.makedirs('dataset/labels/val', exist_ok=True)
os.makedirs('runs/train', exist_ok=True)

data_yaml = {
    'path': os.path.abspath('dataset'),
    'train': 'images/train', 
    'val': 'images/val',
    'names': {
        0: 'fire' 
    }
}

with open('dataset/data.yaml', 'w') as f:
    yaml.dump(data_yaml, f)

def train_yolo_model(model_size='n', epochs=100, batch_size=16, imgsz=640):
    """
    Args:
        model_size: YOLOv8 model size ('n', 's', 'm', 'l', 'x')
        epochs: Number of training epochs
        batch_size: Batch size for training
        imgsz: Image size for training
    """
    model = YOLO(f'yolov8{model_size}.pt') 
    
    results = model.train(
        data='dataset/data.yaml',
        epochs=epochs,
        batch=batch_size,
        imgsz=imgsz,
        patience=20, 
        save=True, 
        device='cpu',
        project='runs/train',
        name=f'yolov8{model_size}_fire_detection'
    )
    
    metrics = model.val()
    print(f"Validation metrics: {metrics}")
    
    return model, results

def predict_with_model(model_path, image_path, conf=0.25):
    model = YOLO(model_path)
    results = model(image_path, conf=conf)
    return results

if __name__ == "__main__":
    model, training_results = train_yolo_model(model_size='n', epochs=50)
    
    # 2. If you already have a trained model, you can load and use it:
    # trained_model_path = 'runs/train/yolov8n_fire_detection/weights/best.pt'
    # result = predict_with_model(trained_model_path, 'path/to/test/image.jpg')
    # result[0].show()  # Display prediction results