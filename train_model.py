"""
Train the skin disease CNN on HAM10000 and save models/skin_model.h5.

Auto-detects dataset under data/raw or common local paths (e.g. Downloads/dataset).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Training short labels (flow_from_dataframe) -> app.py recommendation keys
LABEL_MAP = {
    "nv": "Nevi",
    "mel": "Melanoma",
    "bkl": "Keratosis",
    "bcc": "Carcinoma",
    "akiec": "Actinic",
    "vasc": "Vascular",
    "df": "Dermatofibroma",
}

TRAINING_TO_APP = {
    "Actinic": "Actinic keratoses",
    "Carcinoma": "Basal cell carcinoma",
    "Keratosis": "Benign keratosis",
    "Dermatofibroma": "Dermatofibroma",
    "Melanoma": "Melanoma",
    "Nevi": "Melanocytic nevi",
    "Vascular": "Vascular lesions",
}


def find_dataset(project_root: str) -> tuple[str, str]:
    candidates = [
        (
            os.path.join(project_root, "data", "raw", "HAM10000_metadata.csv"),
            os.path.join(project_root, "data", "raw", "all_images"),
        ),
        (
            r"C:\Users\venka\Downloads\dataset\HAM10000_metadata.csv",
            r"C:\Users\venka\Downloads\dataset",
        ),
        (
            r"C:\Users\venka\Downloads\PDD\ml_server\dataset\HAM10000_metadata.csv",
            r"C:\Users\venka\Downloads\PDD\ml_server\dataset",
        ),
    ]
    for csv_path, image_dir in candidates:
        if os.path.isfile(csv_path) and os.path.isdir(image_dir):
            return csv_path, image_dir
    raise FileNotFoundError(
        "HAM10000 dataset not found. Place HAM10000_metadata.csv and images under "
        "data/raw/ or Downloads/dataset/ (see README)."
    )


def load_dataframe(csv_path: str, image_dir: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    print(f"Loaded metadata: {len(df)} rows")
    print("Indexing images (may take a minute)...")
    all_jpg = glob.glob(os.path.join(image_dir, "**", "*.jpg"), recursive=True)
    path_dict = {os.path.basename(p).split(".")[0]: p for p in all_jpg}
    df["path"] = df["image_id"].map(path_dict)
    missing = int(df["path"].isnull().sum())
    df = df.dropna(subset=["path"])
    df["label"] = df["dx"].map(LABEL_MAP)
    df = df.dropna(subset=["label"])
    print(f"Linked {len(df)} images ({missing} missing from disk)")
    print(df["label"].value_counts())
    return df


def build_model() -> Sequential:
    model = Sequential(
        [
            Conv2D(32, (3, 3), activation="relu", input_shape=(224, 224, 3)),
            MaxPooling2D(2, 2),
            Conv2D(64, (3, 3), activation="relu"),
            MaxPooling2D(2, 2),
            Conv2D(128, (3, 3), activation="relu"),
            MaxPooling2D(2, 2),
            Flatten(),
            Dense(128, activation="relu"),
            Dropout(0.5),
            Dense(7, activation="softmax"),
        ]
    )
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def save_class_labels(class_indices: dict[str, int], out_path: str) -> list[str]:
    ordered_training = [k for k, _ in sorted(class_indices.items(), key=lambda x: x[1])]
    app_labels = [TRAINING_TO_APP[name] for name in ordered_training]
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(app_labels, f, indent=2)
    return app_labels


def main() -> None:
    parser = argparse.ArgumentParser(description="Train skin disease classifier")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--csv", type=str, default="")
    parser.add_argument("--images", type=str, default="")
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.abspath(__file__))
    os.chdir(project_root)

    if args.csv and args.images:
        csv_path, image_dir = args.csv, args.images
    else:
        csv_path, image_dir = find_dataset(project_root)

    df = load_dataframe(csv_path, image_dir)
    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["label"]
    )

    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        vertical_flip=True,
    )
    test_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_loader = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        x_col="path",
        y_col="label",
        target_size=(224, 224),
        batch_size=args.batch_size,
        class_mode="categorical",
        shuffle=True,
    )
    test_loader = test_datagen.flow_from_dataframe(
        dataframe=test_df,
        x_col="path",
        y_col="label",
        target_size=(224, 224),
        batch_size=args.batch_size,
        class_mode="categorical",
        shuffle=False,
    )

    models_dir = os.path.join(project_root, "models")
    os.makedirs(models_dir, exist_ok=True)
    app_labels = save_class_labels(
        train_loader.class_indices,
        os.path.join(models_dir, "class_labels.json"),
    )
    print("Model output order (app labels):", app_labels)

    model = build_model()
    steps_train = max(1, train_loader.samples // args.batch_size)
    steps_val = max(1, test_loader.samples // args.batch_size)

    print(f"\nTraining for {args.epochs} epochs on CPU (this may take 30–90+ minutes)...")
    history = model.fit(
        train_loader,
        steps_per_epoch=steps_train,
        validation_data=test_loader,
        validation_steps=steps_val,
        epochs=args.epochs,
    )

    model_path = os.path.join(models_dir, "skin_model.h5")
    model.save(model_path)
    print(f"\nSaved model: {model_path}")
    print(f"Final val accuracy: {history.history['val_accuracy'][-1]:.4f}")

    # Save training curves for /metrics page
    try:
        import matplotlib.pyplot as plt

        static_dir = os.path.join(project_root, "static")
        os.makedirs(static_dir, exist_ok=True)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        ax1.plot(history.history["accuracy"], label="train")
        ax1.plot(history.history["val_accuracy"], label="val")
        ax1.set_title("Accuracy")
        ax1.legend()
        ax2.plot(history.history["loss"], label="train")
        ax2.plot(history.history["val_loss"], label="val")
        ax2.set_title("Loss")
        ax2.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(static_dir, "training_report.png"))
        plt.close()
        print("Saved static/training_report.png")
    except Exception as e:
        print(f"Could not save training plot: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
