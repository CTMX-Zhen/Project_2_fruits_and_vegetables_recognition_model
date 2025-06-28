
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
    #     - Test these 3 unseen images (real life images) that downloaded from Google
    #     - To check overall accuracy of the best model

    # 6. Platform: offline AI tools & code your own solution
    #     - Visual Studio Code
    #     - Jupyter Notebook
    #     - Python
    #     - pandas, matplotlib, seaborn, scikit-learn, tensorflow, tqdm, opencv-python, icrawler

    # GitHub: https://github.com/CTMX-Zhen/Project_2_fruits_and_vegetables_recognition_model_CTMX
    
"""
    STUDENT'S OUTCOMES
"""
# pip install pandas, matplotlib, seaborn, scikit-learn, tensorflow, tqdm, opencv-python, icrawler

# == IMPORT LIBRARIES ==
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import sys
import logging
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_auc_score, roc_curve

# == HELPER FUNCTIONS ==
# 1. Setup logger
def setup_logger(log_file="model_evaluation.txt"):
    """
    Sets up a logger with two handlers: a file handler and a console handler.
    
    Args:
        log_file (str, optional): The file to write logs to. Defaults to "train_log.txt".
    
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

# 2. Load dataset
# == Load Model Safely ==
def load_model_safely(path, custom_loss=None):
    """
    Loads a Keras model from a file path while safely handling custom loss functions.

    Args:
        path (str): The path to the model file.
        custom_loss (function, optional): A custom loss function to use when loading the model.

    Returns:
        tf.keras.Model: The loaded Keras model.

    """
    # Check if the model has a custom loss function
    if custom_loss:
        # Load the model with the custom loss function
        return tf.keras.models.load_model(path, custom_objects={'loss': custom_loss})
    else:
        # Load the model without a custom loss function
        return tf.keras.models.load_model(path)

# 3. Label Smoothing
def sparse_categorical_crossentropy_with_label_smoothing(n_classes, smoothing=0.1):
    """
    Creates a custom loss function for sparse categorical crossentropy with label smoothing.

    Args:
        n_classes (int): The number of classes in the output.
        smoothing (float, optional): The amount of smoothing to apply. Defaults to 0.1.

    Returns:
        callable: A callable loss function that applies label smoothing to the sparse categorical crossentropy loss.
    """
    loss_fn = tf.keras.losses.CategoricalCrossentropy(label_smoothing=smoothing)

    def loss(y_true, y_pred):
        """
        Computes the label smoothed sparse categorical crossentropy loss.

        Args:
            y_true (tf.Tensor): The true labels.
            y_pred (tf.Tensor): The predicted probabilities.

        Returns:
            tf.Tensor: The label smoothed sparse categorical crossentropy loss.
        """
        y_true_one_hot = tf.one_hot(tf.cast(y_true, tf.int32), depth=n_classes)
        return loss_fn(y_true_one_hot, y_pred)

    return loss

# 4. Best Model Evaluate and Report
def full_model_evaluation(model, test_ds, class_names, model_name="Model", save_dir="."):
    """
    Evaluates the given model on the test dataset and saves the results to the specified directory.

    Args:
        model (tf.keras.Model): The model to evaluate.
        test_ds (tf.data.Dataset): The test dataset to evaluate on.
        class_names (list): A list of class names in the order that they appear in the model's output.
        model_name (str, optional): The name of the model to display in the results. Defaults to "Model".
        save_dir (str, optional): The directory to save the results to. Defaults to ".".

    Returns:
        tuple: A tuple containing the true labels and the predicted labels.
    """
    os.makedirs(save_dir, exist_ok=True)

    y_true, y_pred, y_prob = [], [], []

    for images, labels in test_ds:
        # Predict the output probabilities for the current batch of images
        preds = model.predict(images, verbose=0)

        # Append the labels and predictions to the lists
        y_true.extend(labels.numpy())
        y_pred.extend(np.argmax(preds, axis=1))
        y_prob.extend(preds)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)

    num_classes = len(class_names)

    # === Confusion Matrix ===
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(xticks_rotation=90)
    plt.title(f"{model_name} - Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "confusion_matrix.png"))
    plt.show()

    # === Classification Report ===
    logging.info(f"=== {model_name} - Classification Report ===")
    report_str = classification_report(y_true, y_pred, target_names=class_names)
    logger.info(f"\n{report_str}")

    # === ROC AUC Score ===
    try:
        # Convert the true labels to one-hot encoding
        y_true_onehot = tf.keras.utils.to_categorical(y_true, num_classes)

        # Calculate the ROC AUC score
        auc_score = roc_auc_score(y_true_onehot, y_prob, average="macro", multi_class="ovr")
        logging.info(f"{model_name} - ROC AUC Score (macro, ovr): {auc_score:.4f}")

        # Save the AUC score to a text file
        with open(os.path.join(save_dir, "roc_auc_score.txt"), "w") as f:
            f.write(f"{model_name} - ROC AUC Score (macro, ovr): {auc_score:.4f}\n")

    except Exception as e:
        logging.warning(f"{model_name} - ROC AUC score could not be calculated: {e}")

    # === ROC Curve ===
    try:
        # Convert the true labels to one-hot encoding
        y_true_onehot = tf.keras.utils.to_categorical(y_true, num_classes)

        plt.figure()
        for i in range(num_classes):
            # Calculate the ROC curve for the current class
            fpr, tpr, _ = roc_curve(y_true_onehot[:, i], y_prob[:, i])
            plt.plot(fpr, tpr, label=f"{class_names[i]}")

        # Plot the random chance line
        plt.plot([0, 1], [0, 1], 'k--')

        # Set the axis labels and title
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"{model_name} - ROC Curve")
        plt.legend(loc='lower right', fontsize='small', ncol=2)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "roc_curve.png"))
        plt.show()
    except Exception as e:
        logging.warning(f"{model_name} - ROC curve could not be plotted: {e}")

    # === Per-Class Accuracy Summary ===
    df = pd.DataFrame({
        "true_class": [class_names[i] for i in y_true],
        "pred_class": [class_names[i] for i in y_pred]
    })

    overall_acc = (df["true_class"] == df["pred_class"]).mean()
    logging.info(f"{model_name} - Overall Accuracy: {overall_acc:.2%}")

    summary = df.groupby("true_class").apply(
        lambda g: pd.Series({
            "total": len(g),
            "correct": (g["true_class"] == g["pred_class"]).sum(),
            "accuracy": (g["true_class"] == g["pred_class"]).mean()
        })
    ).reset_index()

    logging.info(f"{model_name} - Per-class Summary:\n" + summary.to_string(index=False))

    # Save chart
    summary_sorted = summary.sort_values("true_class")
    plt.figure(figsize=(20, 6))
    plt.bar(summary_sorted["true_class"], summary_sorted["accuracy"], color="skyblue")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.title(f"Per-Class Accuracy - {model_name}")
    plt.xticks(rotation=75, fontsize=6, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "class_accuracy.png"), dpi=300)
    plt.show()

    # Save CSV
    summary.to_csv(os.path.join(save_dir, "class_accuracy_summary.csv"), index=False)
    logging.info(f"{model_name} - Saved per-class accuracy chart and CSV.")

    return y_true, y_pred

# == START ==
if __name__ == "__main__":
    # Set up logging
    save_dir = "EfficientNetB0-based_image_classifier"
    os.makedirs(save_dir, exist_ok=True)
    logger = setup_logger(log_file=os.path.join(save_dir, "models_evaluation.txt"))

    # == CONFIG ==
    IMG_SIZE = (224, 224)
    BATCH_SIZE = 32
    DATA_DIR = "dataset\\test"
    CLASS_NAMES = sorted(os.listdir(DATA_DIR))
    NUM_CLASSES = len(CLASS_NAMES)

    # == Load Test Dataset ==
    logging.info("Starting dataset loading...")
    test_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False,
        label_mode='int'
    )
    test_ds = test_ds.prefetch(tf.data.AUTOTUNE)

    # == Load and Evaluate All Models ==
    base_model = os.path.join(save_dir, "1. base_model", "base_model.keras")
    fine_tuned_model = os.path.join(save_dir, "2. fine_tuned_model", "fine_tuned_model.keras")
    data_aug_model = os.path.join(save_dir, "3. data_aug_mode", "best_model.keras")

    models_info = [
        ("Baseline Model", base_model, None, "1. base_model"),
        ("Fine-tuned Model", fine_tuned_model, None, "2. fine_tuned_model"),
        ("Data Augmentation Model", data_aug_model, sparse_categorical_crossentropy_with_label_smoothing(NUM_CLASSES, smoothing=0.1), "3. data_aug_mode")
    ]
    
    # Evaluate each model
    logger.info("Evaluating models...")
    for name, path, custom_loss, folder_name in models_info:
        if os.path.exists(path):
            logger.info(f"Loading {name}...")
            model = load_model_safely(path, custom_loss)

            # Make subdirectory for this model's evaluation (same as the model's folder)
            model_save_dir = os.path.join(save_dir, folder_name)
            os.makedirs(model_save_dir, exist_ok=True)

            full_model_evaluation(
                model=model,
                test_ds=test_ds,
                class_names=CLASS_NAMES,
                model_name=name,
                save_dir=model_save_dir
            )
        else:
            logger.warning(f"\nModel {name} not found at path: {path}")

    logging.info("All done!")

# == END ==