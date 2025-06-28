
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

import io
import sys
import logging
import tensorflow as tf
import matplotlib.pyplot as plt

from tqdm.keras import TqdmCallback

# == IMPORT TENSORFLOW LIBRARIES ==
tensorflow_utils = tf.keras.utils
tensorflow_models = tf.keras.models
tensorflow_layers = tf.keras.layers
tensorflow_applications = tf.keras.applications
tensorflow_callback = tf.keras.callbacks.Callback

# == LOGGING CALLBACK ==
class LoggingCallback(tensorflow_callback):
    def on_epoch_end(self, epoch, logs=None):
        """
        This method is called at the end of each epoch during training.
        
        Args:
            epoch (int): The index of the epoch.
            logs (dict, optional): A dictionary containing the training and validation metrics.
        
        Logs:
            The training and validation loss and accuracy for the epoch.
        """
        # Initialize logs if not provided
        logs = logs or {}
        
        # Log the epoch number
        logging.info(f"Epoch {epoch + 1}: ")
        
        # Log the training loss and accuracy
        logging.info(f"loss={logs.get('loss'):.4f}, accuracy={logs.get('accuracy'):.4f}, ")
        
        # Log the validation loss and accuracy
        logging.info(f"val_loss={logs.get('val_loss'):.4f}, val_accuracy={logs.get('val_accuracy'):.4f}")

# == TQDM CALLBACK ==
class TqdmToLogger(io.StringIO):
    def __init__(self, logger, level=logging.INFO):
        """
        Initialize a TqdmToLogger object.
        
        Args:
            logger (logging.Logger): The logger object to write to.
            level (int, optional): The logging level. Defaults to logging.INFO.
        """
        super().__init__()
        self.logger = logger
        self.level = level
        self.buffer = ""

    def write(self, buf):
        """
        Write buffer content to the logger if newline is found.

        Args:
            buf (str): The string buffer to write to the logger.
        """
        # Append buffer content
        self.buffer += buf
        
        # Check for newline character in the buffer
        if "\n" in self.buffer:
            # Split buffer into lines and log each non-empty line
            for line in self.buffer.splitlines():
                if line.strip():  # Log only if line is not empty
                    self.logger.log(self.level, line.strip())
            
            # Clear the buffer
            self.buffer = ""

    def flush(self):
        """
        Flushes the buffer content to the logger if it is not empty.
        
        This method checks if the buffer contains any non-whitespace characters.
        If so, it logs the content and then clears the buffer.
        """
        # Check if buffer contains non-whitespace characters
        if self.buffer.strip():
            # Log the buffer content
            self.logger.log(self.level, self.buffer.strip())
        
        # Clear the buffer
        self.buffer = ""

# == HELPER FUNCTIONS ==
# 1. Setup logger
def setup_logger(log_file="fine_tune_log.txt"):
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
def load_datasets_from_directory(base_dir, img_size=(224, 224), batch_size=32):
    """
    Load train, validation, and test datasets from directory structure.

    The directory structure:
        base_dir/
            train/
                class1/
                class2/
            validation/
                class1/
                class2/
            test/
                class1/
                class2/

    Args:
        base_dir (str): Path to the base dataset directory.
        img_size (tuple): Image resizing dimensions.
        batch_size (int): Batch size for datasets.

    Returns:
        tuple: (train_ds, val_ds, test_ds, class_names)
    """
    logging.info("Loading datasets from directory structure...")

    train_dir = os.path.join(base_dir, 'train')
    val_dir = os.path.join(base_dir, 'validation')
    test_dir = os.path.join(base_dir, 'test')

    train_ds = tensorflow_utils.image_dataset_from_directory(
        train_dir,
        image_size=img_size,
        batch_size=batch_size,
        shuffle=True,
        label_mode='int'
    )
    val_ds = tensorflow_utils.image_dataset_from_directory(
        val_dir,
        image_size=img_size,
        batch_size=batch_size,
        shuffle=False,
        label_mode='int'
    )
    test_ds = tensorflow_utils.image_dataset_from_directory(
        test_dir,
        image_size=img_size,
        batch_size=batch_size,
        shuffle=False,
        label_mode='int'
    )

    class_names = train_ds.class_names
    logging.info(f"Classes: {class_names}")

    # Prefetch datasets for performance
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)
    test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds, test_ds, class_names

# 3. Build and compile model
def build_model(input_shape, num_classes, base_trainable=False, freeze_until=None):
    """
    Builds a transfer learning model using EfficientNetB0 as the backbone.

    Args:
        input_shape (tuple): The shape of the input images (height, width, channels).
        num_classes (int): The number of output classes.
        base_trainable (bool): If True, unfreezes the base EfficientNet model for fine-tuning.
        freeze_until (int or None): If base_trainable is True, freeze layers up to this index.

    Returns:
        tensorflow.keras.Model: A compiled Keras model.
    """
    # Load base EfficientNetB0 model (exclude top classification head)
    base_model = tensorflow_applications.EfficientNetB0(
        include_top=False,
        weights='imagenet',
        input_shape=input_shape
    )

    # Set trainability
    base_model.trainable = base_trainable

    if base_trainable and freeze_until is not None:
        # Optionally freeze first `freeze_until` layers
        for layer in base_model.layers[:freeze_until]:
            layer.trainable = False
        for layer in base_model.layers[freeze_until:]:
            layer.trainable = True

    # Add custom classification head
    x = base_model.output
    x = tensorflow_layers.GlobalAveragePooling2D()(x)
    x = tensorflow_layers.Dropout(0.3)(x)
    outputs = tensorflow_layers.Dense(num_classes, activation='softmax')(x)

    # Construct the full model
    model = tensorflow_models.Model(inputs=base_model.input, outputs=outputs)

    # Compile the model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4 if not base_trainable else 1e-5),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

# 4. Train model
def plot_history(history):
    """
    Plots training and validation metrics from a history object.

    Args:
        history (dict): A dictionary containing the training and validation metrics.
    """
    # Get metrics from history object
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    # Plot metrics
    plt.figure(figsize=(8, 6))
    plt.subplot(1, 2, 1)
    plt.plot(acc, label='Train Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Accuracy')

    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(loss, label='Train Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Loss')

    # Save and show plot
    plt.savefig(os.path.join(save_dir, "fine_tune_result.png"))
    plt.show()

# == START ==
if __name__ == "__main__":
    # Set up logging
    save_dir = "EfficientNetB0-based_image_classifier\\2. fine_tuned_model"
    os.makedirs(save_dir, exist_ok=True)
    logger = setup_logger(log_file=os.path.join(save_dir, "fine_tune_log.txt"))

    # === CONFIG ===
    data_dir = "dataset"
    img_size = (224, 224)
    batch_size = 32
    fine_tune_epochs = 5  # typically 5–10, already trained 10 epochs in stage 1

    # Load dataset
    logging.info("Starting dataset loading...")
    train_ds, val_ds, test_ds, class_names = load_datasets_from_directory(
        base_dir=data_dir, img_size=img_size, batch_size=batch_size
    )

    # === FINE-TUNING ===
    # Build and compile model
    logging.info("Building model for fine-tuning...")
    model = build_model(
        input_shape=(img_size[0], img_size[1], 3),
        num_classes=len(class_names),
        base_trainable=True,
        freeze_until=150  # freeze first 150 layers of EfficientNetB0
    )

    # Load previously trained weights (optional but recommended)
    if os.path.exists("EfficientNetB0-based_image_classifier\\1. base_model\\base_model.keras"):
        logging.info("Loading weights from base model...")
        model.load_weights("EfficientNetB0-based_image_classifier\\1. base_model\\base_model.keras")
    
    # Compile with a smaller learning rate for fine-tuning
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # Print model summary
    stream = io.StringIO()
    model.summary(print_fn=lambda x: stream.write(x + "\n"))
    summary_str = stream.getvalue()
    summary_str_ascii = summary_str.encode('ascii', errors='ignore').decode('ascii')
    logging.info("\n" + summary_str_ascii)

    # Train (fine-tune) model
    logging.info("Starting fine-tuning...")
    history_finetune = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=fine_tune_epochs,
        callbacks=[LoggingCallback(), TqdmCallback(file=TqdmToLogger(logger))]
    )

    # Evaluate model on test set
    test_loss, test_acc = model.evaluate(test_ds)
    logging.info(f"Test Accuracy after fine-tuning: {test_acc:.4f}")

    # Save fine tuned model
    logging.info("Training completed. Saving model...")
    model.save(os.path.join(save_dir, "fine_tuned_model.keras"))

    # Plot training history
    logging.info("Plotting training history...")
    plot_history(history_finetune)

    logging.info("All done!")

# == END ==