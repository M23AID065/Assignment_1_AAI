# CNN-TCN Jet Detection

This project detects multiple targets (Alien/Friendly) in image streams using a hybrid CNN-TCN deep learning model.

## Folders:
- `model/`: training and evaluation scripts.
- `inference/`: prediction and visualization on new images.

## How to Run
```bash
# Train
python model/train_model.py

# Inference
python inference/detect_in_frame.py
```

Ensure you place test images in the `inference/` folder.

## Requirements
- TensorFlow
- keras-tcn
- OpenCV
- matplotlib, seaborn, sklearn
