"""
Train the skin disease classifier on HAM10000 using transfer learning.

Uses EfficientNetB0 (ImageNet pretrained) with class balancing and fine-tuning
to target ~90% validation accuracy.
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
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.applications.efficientnet import EfficientNetB0, preprocess_input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

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

CHART_LABELS = ["AK", "BCC", "DF", "BKL", "MEL", "NV", "VASC"]


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


def build_model(fine_tune: bool = False) -> tuple[Model, Model]:
    base = EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(224, 224, 3),
        pooling=None,
    )
    base.trainable = fine_tune
    if fine_tune:
        for layer in base.layers[:-40]:
            layer.trainable = False

    x = GlobalAveragePooling2D(name="gap")(base.output)
    x = Dropout(0.4)(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.3)(x)
    outputs = Dense(7, activation="softmax")(x)
    model = Model(inputs=base.input, outputs=outputs, name="skin_classifier")
    return model, base


def class_weight_dict(train_df: pd.DataFrame, class_indices: dict[str, int]) -> dict[int, float]:
    unique = np.unique(train_df["label"])
    weights = compute_class_weight("balanced", classes=unique, y=train_df["label"])
    return {class_indices[label]: float(w) for label, w in zip(unique, weights)}


def save_class_labels(class_indices: dict[str, int], out_path: str) -> list[str]:
    ordered_training = [k for k, _ in sorted(class_indices.items(), key=lambda x: x[1])]
    app_labels = [TRAINING_TO_APP[name] for name in ordered_training]
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(app_labels, f, indent=2)
    return app_labels


def merge_histories(*histories) -> dict:
    merged: dict[str, list] = {}
    for h in histories:
        for key, values in h.history.items():
            merged.setdefault(key, []).extend(values)
    return merged


def save_training_plot(history: dict, static_dir: str) -> None:
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history["accuracy"], label="train")
    ax1.plot(history["val_accuracy"], label="val")
    ax1.axhline(0.9, color="gray", linestyle="--", linewidth=0.8, label="90% target")
    ax1.set_title("Accuracy")
    ax1.set_ylim(0, 1)
    ax1.legend()
    ax2.plot(history["loss"], label="train")
    ax2.plot(history["val_loss"], label="val")
    ax2.set_title("Loss")
    ax2.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(static_dir, "training_report.png"))
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train skin disease classifier (EfficientNet)")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--head-epochs", type=int, default=8)
    parser.add_argument("--finetune-epochs", type=int, default=15)
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
    train_df, val_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["label"]
    )

    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=25,
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.1,
        zoom_range=0.15,
        horizontal_flip=True,
        vertical_flip=True,
        brightness_range=(0.85, 1.15),
    )
    val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    train_loader = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        x_col="path",
        y_col="label",
        target_size=(224, 224),
        batch_size=args.batch_size,
        class_mode="categorical",
        shuffle=True,
    )
    val_loader = val_datagen.flow_from_dataframe(
        dataframe=val_df,
        x_col="path",
        y_col="label",
        target_size=(224, 224),
        batch_size=args.batch_size,
        class_mode="categorical",
        shuffle=False,
    )

    models_dir = os.path.join(project_root, "models")
    static_dir = os.path.join(project_root, "static")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(static_dir, exist_ok=True)

    app_labels = save_class_labels(
        train_loader.class_indices,
        os.path.join(models_dir, "class_labels.json"),
    )
    with open(os.path.join(models_dir, "model_config.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "backbone": "efficientnetb0",
                "input_size": 224,
                "preprocess": "efficientnet",
                "chart_labels": CHART_LABELS,
            },
            f,
            indent=2,
        )
    print("Model output order (app labels):", app_labels)

    class_weights = class_weight_dict(train_df, train_loader.class_indices)
    print("Class weights:", class_weights)

    model_path = os.path.join(models_dir, "skin_model.h5")
    checkpoint = ModelCheckpoint(
        model_path,
        monitor="val_accuracy",
        mode="max",
        save_best_only=True,
        verbose=1,
    )
    early_stop = EarlyStopping(
        monitor="val_accuracy",
        mode="max",
        patience=6,
        restore_best_weights=True,
        verbose=1,
    )
    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.3,
        patience=3,
        min_lr=1e-7,
        verbose=1,
    )

    steps_train = max(1, train_loader.samples // args.batch_size)
    steps_val = max(1, val_loader.samples // args.batch_size)

    # Phase 1: train classifier head (frozen backbone)
    print("\n=== Phase 1: Training classifier head (frozen EfficientNet) ===")
    model, base = build_model(fine_tune=False)
    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    history1 = model.fit(
        train_loader,
        steps_per_epoch=steps_train,
        validation_data=val_loader,
        validation_steps=steps_val,
        epochs=args.head_epochs,
        class_weight=class_weights,
        callbacks=[checkpoint, reduce_lr],
        verbose=1,
    )

    best_val = max(history1.history["val_accuracy"])
    print(f"Phase 1 best val accuracy: {best_val:.4f}")

    # Phase 2: fine-tune top layers on the same trained model
    print("\n=== Phase 2: Fine-tuning top EfficientNet layers ===")
    base.trainable = True
    for layer in base.layers[:-40]:
        layer.trainable = False
    model.compile(
        optimizer=Adam(learning_rate=1e-5),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    history2 = model.fit(
        train_loader,
        steps_per_epoch=steps_train,
        validation_data=val_loader,
        validation_steps=steps_val,
        epochs=args.finetune_epochs,
        class_weight=class_weights,
        callbacks=[checkpoint, early_stop, reduce_lr],
        verbose=1,
    )

    merged = merge_histories(history1, history2)
    best_val = max(merged["val_accuracy"])
    print(f"\nBest validation accuracy: {best_val:.4f} ({best_val * 100:.1f}%)")

    # Reload best checkpoint weights
    if os.path.isfile(model_path):
        from tensorflow.keras.models import load_model

        model = load_model(model_path)
        model.save(model_path)

    try:
        save_training_plot(merged, static_dir)
        print("Saved static/training_report.png")
    except Exception as e:
        print(f"Could not save training plot: {e}", file=sys.stderr)

    if best_val >= 0.9:
        print("Target reached: validation accuracy >= 90%")
    else:
        print(
            f"Val accuracy is {best_val * 100:.1f}%. "
            "Try re-running with --finetune-epochs 25 for more gains."
        )


if __name__ == "__main__":
    main()
