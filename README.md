Here’s a clean and professional `README.md` for your fire detection project using YOLOv8 and Ultralytics:

---

```markdown
# 🔥 Fire Detection Using YOLOv8

This project uses the [Ultralytics YOLOv8](https://docs.ultralytics.com) model to detect **fire** in images. It sets up a YOLO-compatible dataset structure, trains a custom object detection model, and provides utilities for inference.

## 📁 Project Structure

```

project/
│
├── dataset/
│   ├── images/train/   # Training images
│   ├── images/val/     # Validation images
│   ├── labels/train/   # Training labels (YOLO format)
│   ├── labels/val/     # Validation labels
│   └── data.yaml       # YOLO dataset config file
│
├── runs/train/         # Training logs and saved models
├── main.py             # Main training and inference script
├── requirements.txt    # Python dependencies (generate using pipreqs)
└── README.md

````

## 🚀 Features

- Auto-creates dataset directories and a YOLO-compatible `data.yaml`
- Trains a YOLOv8 model on a custom fire dataset
- Provides a prediction function to test the trained model
- Runs on CPU by default

## 🧠 Model Training

To train the YOLOv8 model (e.g., YOLOv8n):

```bash
python main.py
````

This will:

* Create necessary folders
* Generate `dataset/data.yaml`
* Train a YOLOv8n model for 50 epochs

## 📸 Inference Example

Uncomment and modify the following lines in `main.py` to test the model:

```python
# trained_model_path = 'runs/train/yolov8n_fire_detection/weights/best.pt'
# result = predict_with_model(trained_model_path, 'path/to/test/image.jpg')
# result[0].show()  # Show image with detections
```

## ⚙️ Configuration

You can customize training in the `train_yolo_model` function:

```python
train_yolo_model(model_size='n', epochs=50, batch_size=16, imgsz=640)
```

* `model_size`: Choose from `'n'`, `'s'`, `'m'`, `'l'`, `'x'`
* `epochs`: Training epochs
* `batch_size`: Images per batch
* `imgsz`: Input image size

## 📦 Installation

Make sure you have Python 3.8+ installed. Then install dependencies:

```bash
pip install ultralytics PyYAML
```

Or generate a `requirements.txt` with:

```bash
pipreqs . --force
pip install -r requirements.txt
```

## 📝 License

This project is licensed under the MIT License.

---

## 🙋‍♂️ Contact

For questions or suggestions, feel free to open an issue or fork the repo and contribute!

```

---

Let me know if you'd like to add badges (build status, license, etc.) or adapt this for Colab or GPU support.
```
