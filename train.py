#!/usr/bin/env python3
"""
Train YOLOv11 on mosquito dataset
"""

import os
from ultralytics import YOLO

# ------------------------------
# 1️⃣ Paths and configuration
# ------------------------------

# Path to your dataset yaml file
DATA_YAML = '/home/mosquitno/Desktop/mosquitno/dataset/data.yaml'

# Pretrained YOLOv11 model (nano version for Raspberry Pi)
PRETRAINED_MODEL = YOLO("yolo11n.pt")

# Training parameters
IMG_SIZE = 640        # Input image size
BATCH_SIZE = 4        # Small batch for Raspberry Pi CPU
EPOCHS = 8           # Number of epochs
PROJECT_NAME = 'mosquito_train'
EXPERIMENT_NAME = 'exp1'  # Change if re-running

# ------------------------------
# 2️⃣ Initialize YOLO model
# ------------------------------

model = YOLO(PRETRAINED_MODEL)

# ------------------------------
# 3️⃣ Start training
# ------------------------------

model.train(
    data=DATA_YAML,        # path to data.yaml
    imgsz=IMG_SIZE,        # input image size
    batch=BATCH_SIZE,      # batch size
    epochs=EPOCHS,         # number of training epochs
    project=PROJECT_NAME,  # folder to save results
    name=EXPERIMENT_NAME,  # experiment name
    device='cpu',          # 'cpu' for Raspberry Pi (if no GPU)
    workers=2,             # number of dataloader workers
    cache=False,           # avoid caching images on Pi to save RAM
    augment=True           # apply data augmentation
)

# ------------------------------
# 4️⃣ Optional: Evaluate
# ------------------------------

metrics = model.val(
    data=DATA_YAML,
    imgsz=IMG_SIZE,
    batch=BATCH_SIZE
)

print("Validation metrics:", metrics)
