
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

import cv2
import numpy as np
import tensorflow as tf
import tkinter as tk
from tkinter import filedialog, messagebox
import matplotlib.pyplot as plt

# == CONFIGURATION ==
MODEL_PATH = "EfficientNetB0-based_image_classifier\\3. data_aug_mode\\best_model.keras"
IMG_SIZE = (224, 224)
CLASS_NAMES = sorted(os.listdir("dataset\\train"))
NUM_CLASSES = len(CLASS_NAMES)

# == Custom loss (only needed for label smoothing model) ==
def sparse_categorical_crossentropy_with_label_smoothing(n_classes, smoothing=0.1):
    loss_fn = tf.keras.losses.CategoricalCrossentropy(label_smoothing=smoothing)
    def loss(y_true, y_pred):
        y_true_one_hot = tf.one_hot(tf.cast(y_true, tf.int32), depth=n_classes)
        return loss_fn(y_true_one_hot, y_pred)
    return loss

# == Load model safely ==
def load_model_safely(path, custom_loss=None):
    if custom_loss:
        return tf.keras.models.load_model(path, custom_objects={'loss': custom_loss})
    return tf.keras.models.load_model(path)

# == Predict a single image using raw pixel values (model handles Rescaling) ==
def predict_single_image_with_model_call(model, image_path, img_size, class_names):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Image not readable: {image_path}")

    img_resized = cv2.resize(img, img_size)
    # DO NOT normalize here since model already includes Rescaling(1./255)
    input_tensor = tf.convert_to_tensor([img_resized], dtype=tf.float32)

    probs = model(input_tensor, training=False).numpy()[0]

    # Show all probabilities sorted
    print("\n=== All Class Probabilities (Top to Bottom) ===")
    sorted_indices = np.argsort(probs)[::-1]
    for i in sorted_indices:
        print(f"{class_names[i]}: {probs[i]:.4f}")

    # Show top-3
    print("\n=== Top-3 Predictions ===")
    for i in sorted_indices[:3]:
        print(f"{class_names[i]}: {probs[i]:.4f}")

    pred_idx = sorted_indices[0]
    pred_class = class_names[pred_idx]
    confidence = float(probs[pred_idx])

    return pred_class, confidence, cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

# == MAIN ==
if __name__ == "__main__":
    print("Loading model...")
    model = load_model_safely(
        MODEL_PATH,
        custom_loss=sparse_categorical_crossentropy_with_label_smoothing(NUM_CLASSES, 0.1)
    )

    print("Select an image...")
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Select an image",
        filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp")]
    )

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

    # Show the result
    plt.imshow(img_rgb)
    plt.axis("off")
    plt.title(f"{pred_class} ({confidence * 100:.2f}%)", fontsize=14)
    plt.tight_layout()
    plt.show()
