
"""
Project 2 Task Description：
- Download Fruits and Vegetables Image Recognition Dataset at: https://www.kaggle.com/datasets/kritikseth/fruit-and-vegetable-image-recognition/data

Task：
- Identify suitable image recognition solution for the dataset
- Implement the solution using any tools or languages (you can refer to  'code' tab at the webpage)
- Analyze the results

Submission:
- Academic journal paper using IEEE format (use word or latex). Paper content includes:
- Introduction to the image recognition problem and your proposed solution.
- Explanation and justification of the proposed solution architecture and parameters
- Complete screenshot and explanation of the implementation
- Result analysis and discussion.
- Complete working source code or implementation files.

NAME: Connie Tang Ming Xin
MATRIC NUMBER: MEC245051
"""

# Project 2
    # 1. Dataset: dataset
    #     - Sourse downloaded from Kaggle: https://www.kaggle.com/datasets/kritikseth/fruit-and-vegetable-image-recognition/data
    #     - Description: 
    #         - 36 classes of different types of fruits and vegetables
    #         - dataset already split into:
    #             - dataset\train
    #             - dataset\validation
    #             - dataset\test

    # 2. Model: EfficientNetB0-based_image_classifier (as the base model)
    #     - Architecture: Transfer learning model using EfficientNetB0 as the backbone.
    #     - Classifier Head: Custom Dense layer with softmax output for multi-class classification.
    #     - Input Size: 224×224×3 RGB images.
    #     - Output Classes: 150 (or however many you have in your dataset).
    #     - Pretrained Weights: ImageNet.
        
    # 3. Fine Tune (use the base model to fine tune)
    #     - Initially trained with EfficientNetB0 frozen (baseline model).
    #     - Then fine-tuned by unfreezing the top layers of EfficientNetB0 for domain-specific feature learning.
    #     - Fine-tuning strategy:
    #         - Unfroze top N layers (adjustable) while keeping others frozen.
    #         - Reduced learning rate (1e-5) to avoid catastrophic forgetting.
    #         - Continued training on same dataset with validation monitoring.

    # 4. Data Augmentation (use the fine tuned model to train with data augmentation)
    #     - Applied only to the training dataset using TensorFlow’s preprocessing layers.
    #     - Augmentation techniques used:
    #         - RandomFlip("horizontal") – horizontal mirroring.
    #         - RandomRotation(0.1) – slight random rotation.
    #         - RandomZoom(0.1) – slight zoom-in effects.
    #         - RandomContrast(0.1) – small contrast adjustments.
    #     - Integrated using .map() before training to apply on-the-fly.
    #     - Aimed to improve generalization and reduce overfitting.
    #     - Label smoothing (ε = 0.1) used in the loss function for better confidence calibration.
    #     - Fine-tuning was done with:
    #         - Partially unfrozen EfficientNetB0 (freeze_until = 150).
    #         - Low learning rate (1e-5).
    #     - Early stopping, learning rate scheduler, and checkpointing were used.
    #     The best model was saved and evaluated on the test dataset.

    # 5. Model Evaluation
    #     - Confusion Matrix
    #     - ROC AUC

    # 6. Use 'icrawler' to download 3 images for each class from Google
    #     - Load the best model trained
    #     - Test these 3 unseen images (real life images) that dowloaded from Google
    #     - To check overall accuracy of the best model

    # 6. Platform: offline AI tools & code your own solution
    #     - Visual Studio Code
    #     - Jupyter Notebook
    #     - Python
    #     - pandas, matplotlib, seaborn, scikit-learn, tensorflow, tqdm, opencv-python, icrawler

    # GitHub: https://github.com/CTMX-Zhen/Project_2_fruits_and_vegetables_recognition_model
    
"""
    STUDENT'S OUTCOMES
"""
# pip install pandas, matplotlib, seaborn, scikit-learn, tensorflow, tqdm, opencv-python, icrawler

# == IMPORT LIBRARIES ==
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import sys
import cv2
import logging

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from tkinter import messagebox
from icrawler.builtin import GoogleImageCrawler

# == HELPER FUNCTIONS ==
# 1. Setup logger
def setup_logger(log_file="test_log.txt"):
    """
    Sets up a logger with two handlers: a file handler and a console handler.

    Args:
        log_file (str, optional): The file to write logs to. Defaults to "test_log.txt".

    Returns:
        logging.Logger: The logger object.
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Formatter
    # The formatter is used to format the log messages.
    # It takes the log message as input and returns a string.
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # File handler
    # The file handler is used to write logs to a file.
    # It takes the log file path as an argument.
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # Console handler
    # The console handler is used to write logs to the console.
    # It takes the output stream as an argument.
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger

# 2. Load class names
def load_class_names(data_dir):
    """
    Loads the class names from a directory.

    Args:
        data_dir (str): The path to the dataset directory.

    Returns:
        list: A list of class names.
    """
    class_names = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    logging.info(f"Found classes: {class_names}")
    return class_names

# 3. Download test images for each class
def download_test_images(class_names, test_dir, images_per_class):
    """
    Downloads a specified number of test images for each class from Google.

    Args:
        class_names (list): A list of class names to download images for.
        test_dir (str): The path to the directory where test images will be stored.
        images_per_class (int): The number of images to download for each class.

    Returns:
        None
    """
    # Create the test directory if it does not exist
    os.makedirs(test_dir, exist_ok=True)

    # Iterate through each class name
    for cls in class_names:
        # Create a directory for the current class if it does not exist
        cls_dir = os.path.join(test_dir, cls)
        os.makedirs(cls_dir, exist_ok=True)

        # Define the search term for the current class
        search_term = f"{cls}"

        # Initialize the GoogleImageCrawler with the storage directory
        crawler = GoogleImageCrawler(storage={"root_dir": cls_dir})

        # Crawl the web for images according to the search term and max number
        crawler.crawl(keyword=search_term, max_num=images_per_class)

        # Log the download completion for the current class
        logging.info(f"Downloaded {images_per_class} images for class: {cls}")

# 4. Label Smoothing
def sparse_categorical_crossentropy_with_label_smoothing(n_classes, smoothing=0.1):
    loss_fn = tf.keras.losses.CategoricalCrossentropy(label_smoothing=smoothing)
    def loss(y_true, y_pred):
        y_true_one_hot = tf.one_hot(tf.cast(y_true, tf.int32), depth=n_classes)
        return loss_fn(y_true_one_hot, y_pred)
    return loss

# 5. Load model
def load_model_safely(path, custom_loss=None):
    if custom_loss:
        return tf.keras.models.load_model(path, custom_objects={'loss': custom_loss})
    return tf.keras.models.load_model(path)

# 5. Prediction
def predict_single_image_with_model_call(model, image_path, img_size, class_names):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Image not readable: {image_path}")

    img_resized = cv2.resize(img, img_size)
    # DO NOT normalize here since model already includes Rescaling(1./255)
    input_tensor = tf.convert_to_tensor([img_resized], dtype=tf.float32)

    probs = model(input_tensor, training=False).numpy()[0]

    # Show all probabilities sorted
    logger.info("\n=== All Class Probabilities (Top to Bottom) ===")
    sorted_indices = np.argsort(probs)[::-1]
    for i in sorted_indices:
        logger.info(f"{class_names[i]}: {probs[i]:.4f}")

    # Show top-3
    logger.info("\n=== Top-3 Predictions ===")
    for i in sorted_indices[:3]:
        logger.info(f"{class_names[i]}: {probs[i]:.4f}")

    pred_idx = sorted_indices[0]
    pred_class = class_names[pred_idx]
    confidence = float(probs[pred_idx])

    return pred_class, confidence, cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

# 7. Summarize and Visualize
def summarize_and_visualize(df):
    """
    Summarizes and visualizes the classification accuracy.

    Args:
        df (DataFrame): DataFrame containing columns 'pred_class' and 'true_class'.

    Returns:
        None
    """
    # Calculate and log overall accuracy
    overall_acc = (df['pred_class'] == df['true_class']).mean()
    logging.info(f"Overall Accuracy: {overall_acc:.2%}")

    # Group by true class and calculate total, correct counts, and accuracy
    summary = df.groupby("true_class").apply(
        lambda g: pd.Series({
            "total": len(g),
            "correct": (g['pred_class'] == g['true_class']).sum(),
            "accuracy": (g['pred_class'] == g['true_class']).mean()
        })
    ).reset_index()

    # Log per-class summary
    logging.info("Per-class Summary:")
    logging.info(f"\n{summary}")

    # Sort by class name for consistent display
    summary = summary.sort_values("true_class")

    # Create a bar chart for per-class accuracy
    plt.figure(figsize=(20, 6))  # Set figure size to fit all labels
    plt.bar(summary["true_class"], summary["accuracy"], color="skyblue")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.title("Per-Class Accuracy (All 150 Classes)")

    # Improve x-axis label readability
    plt.xticks(rotation=75, fontsize=6, ha='right')
    plt.tight_layout()

    # Save the bar chart as an image file
    plt.savefig("dataset\\class_accuracy_full.png", dpi=300)
    plt.show()

    # Save summary as CSV
    summary.to_csv("dataset\\class_accuracy_summary.csv", index=False)
    logging.info("Saved full per-class accuracy chart and CSV summary.")

# == MAIN ==
if __name__ == "__main__":
    # Set up logging
    save_dir = "EfficientNetB0-based_image_classifier"
    os.makedirs(save_dir, exist_ok=True)
    logger = setup_logger(log_file=os.path.join(save_dir, "test_unseen_images.txt"))

    # === CONFIG ===
    MODEL_PATH = "EfficientNetB0-based_image_classifier\\3. data_aug_mode\\best_model.keras"
    IMG_SIZE = (224, 224)
    CLASS_NAMES = sorted(os.listdir("dataset\\train"))
    NUM_CLASSES = len(CLASS_NAMES)

    data_dir = "dataset\\train"
    test_dir = "dataset\\test_unseen_images"
    images_per_class = 3

    # Load class names
    class_names = load_class_names(data_dir)

    # Download test images
    download_test_images(class_names, test_dir, images_per_class)

    logger.info("Loading model...")
    model = load_model_safely(
        MODEL_PATH,
        custom_loss=sparse_categorical_crossentropy_with_label_smoothing(NUM_CLASSES, 0.1)
    )

    logger.info("Predicting all images in test_unseen_images...")

    results = []

    for class_folder in sorted(os.listdir(test_dir)):
        folder_path = os.path.join(test_dir, class_folder)
        if not os.path.isdir(folder_path):
            continue

        for file_name in sorted(os.listdir(folder_path)):
            file_path = os.path.join(folder_path, file_name)
            try:
                pred_class, confidence, img_rgb = predict_single_image_with_model_call(
                    model, file_path, IMG_SIZE, CLASS_NAMES
                )

                # Log or logger.info result
                logging.info(f"Image: {file_path} | Predicted: {pred_class} ({confidence:.2%}) | True: {class_folder}")

                results.append({
                    "filename": file_name,
                    "true_class": class_folder,
                    "pred_class": pred_class,
                    "confidence": confidence
                })

            except Exception as e:
                logging.warning(f"Failed to process {file_path}: {e}")

    if not file_path:
        messagebox.showinfo("Cancelled", "No image was selected.")
        exit()

    try:
        pred_class, confidence, img_rgb = predict_single_image_with_model_call(
            model, file_path, IMG_SIZE, CLASS_NAMES
        )
    except Exception as e:
        messagebox.showerror("Error", str(e))
        exit()
    
    # After collecting all predictions
    df = pd.DataFrame(results)

    # Save the raw predictions CSV
    df.to_csv(os.path.join("dataset", "test_unseen_predictions.csv"), index=False)

    # Summarize and visualize
    summarize_and_visualize(df)

    logging.info("All done!")

# == END ==