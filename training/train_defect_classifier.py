import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import cv2
from PIL import Image
import json
import mlflow
import mlflow.tensorflow
import tempfile
import shutil

class DefectClassifier:
    def __init__(self, image_size=(224, 224), batch_size=32, experiment_name="defect-classification"):
        self.image_size = image_size
        self.batch_size = batch_size
        self.model = None
        self.history = None
        self.experiment_name = experiment_name
        
        # Setup MLflow
        mlflow.set_experiment(experiment_name)
        
    def start_mlflow_run(self, run_name=None):
        """Start an MLflow run"""
        return mlflow.start_run(run_name=run_name)
        
    def create_model(self):
        """Create a CNN model for binary classification"""
        model = keras.Sequential([
            # Input layer
            layers.Input(shape=(*self.image_size, 3)),
            
            # Convolutional layers
            layers.Conv2D(32, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            
            # Flatten and dense layers
            layers.Flatten(),
            layers.Dropout(0.5),
            layers.Dense(512, activation='relu'),
            layers.Dense(1, activation='sigmoid')  # Binary classification
        ])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        return model
    
    def prepare_data(self, data_dir):
        """Prepare training and validation data generators"""
        # Data augmentation for training
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            validation_split=0.2
        )
        
        # Only rescaling for validation
        val_datagen = ImageDataGenerator(
            rescale=1./255,
            validation_split=0.2
        )
        
        train_generator = train_datagen.flow_from_directory(
            data_dir,
            target_size=self.image_size,
            batch_size=self.batch_size,
            class_mode='binary',
            subset='training'
        )
        
        validation_generator = val_datagen.flow_from_directory(
            data_dir,
            target_size=self.image_size,
            batch_size=self.batch_size,
            class_mode='binary',
            subset='validation'
        )
        
        return train_generator, validation_generator
    
    def train(self, data_dir, epochs=10, run_name=None):
        """Train the model with MLflow tracking"""
        if self.model is None:
            self.create_model()
        
        with mlflow.start_run(run_name=run_name):
            # Log parameters
            mlflow.log_param("epochs", epochs)
            mlflow.log_param("batch_size", self.batch_size)
            mlflow.log_param("image_size", self.image_size)
            mlflow.log_param("data_dir", data_dir)
            
            # Log model architecture
            mlflow.log_param("model_layers", len(self.model.layers))
            mlflow.log_param("total_params", self.model.count_params())
            
            train_gen, val_gen = self.prepare_data(data_dir)
            
            # Log data info
            mlflow.log_param("train_samples", train_gen.samples)
            mlflow.log_param("val_samples", val_gen.samples)
            mlflow.log_param("num_classes", train_gen.num_classes)
            mlflow.log_param("class_names", list(train_gen.class_indices.keys()))
            
            # Create MLflow callback
            class MLflowCallback(keras.callbacks.Callback):
                def on_epoch_end(self, epoch, logs=None):
                    if logs:
                        mlflow.log_metric("train_loss", logs.get("loss"), step=epoch)
                        mlflow.log_metric("train_accuracy", logs.get("accuracy"), step=epoch)
                        mlflow.log_metric("val_loss", logs.get("val_loss"), step=epoch)
                        mlflow.log_metric("val_accuracy", logs.get("val_accuracy"), step=epoch)
            
            # Callbacks
            early_stopping = keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=3,
                restore_best_weights=True
            )
            
            model_checkpoint = keras.callbacks.ModelCheckpoint(
                'best_defect_model.h5',
                monitor='val_loss',
                save_best_only=True
            )
            
            mlflow_callback = MLflowCallback()
            
            # Train the model
            self.history = self.model.fit(
                train_gen,
                epochs=epochs,
                validation_data=val_gen,
                callbacks=[early_stopping, model_checkpoint, mlflow_callback]
            )
            
            # Log final metrics
            final_train_loss = self.history.history['loss'][-1]
            final_train_acc = self.history.history['accuracy'][-1]
            final_val_loss = self.history.history['val_loss'][-1]
            final_val_acc = self.history.history['val_accuracy'][-1]
            
            mlflow.log_metric("final_train_loss", final_train_loss)
            mlflow.log_metric("final_train_accuracy", final_train_acc)
            mlflow.log_metric("final_val_loss", final_val_loss)
            mlflow.log_metric("final_val_accuracy", final_val_acc)
            
            # Log model
            mlflow.tensorflow.log_model(self.model, "model")
            
            # Save and log training history plot
            if self.history:
                self.plot_training_history(save_plot=True)
                mlflow.log_artifact("training_history.png")
            
            # Log model summary
            model_summary_path = "model_summary.txt"
            with open(model_summary_path, 'w', encoding='utf-8') as f:
                self.model.summary(print_fn=lambda x: f.write(x + '\n'))
            mlflow.log_artifact(model_summary_path)
            os.remove(model_summary_path)
            
            print(f"MLflow run completed. Run ID: {mlflow.active_run().info.run_id}")
            
        return self.history
    
    def evaluate(self, test_dir, log_to_mlflow=True):
        """Evaluate the model on test data"""
        test_datagen = ImageDataGenerator(rescale=1./255)
        
        test_generator = test_datagen.flow_from_directory(
            test_dir,
            target_size=self.image_size,
            batch_size=self.batch_size,
            class_mode='binary',
            shuffle=False
        )
        
        # Predictions
        predictions = self.model.predict(test_generator)
        predicted_classes = (predictions > 0.5).astype(int)
        
        # True labels
        true_labels = test_generator.classes
        
        # Classification report
        class_names = list(test_generator.class_indices.keys())
        report = classification_report(true_labels, predicted_classes, target_names=class_names)
        
        # Confusion matrix
        cm = confusion_matrix(true_labels, predicted_classes)
        
        if log_to_mlflow and mlflow.active_run():
            # Log test metrics
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            
            accuracy = accuracy_score(true_labels, predicted_classes)
            precision = precision_score(true_labels, predicted_classes, average='weighted')
            recall = recall_score(true_labels, predicted_classes, average='weighted')
            f1 = f1_score(true_labels, predicted_classes, average='weighted')
            
            mlflow.log_metric("test_accuracy", accuracy)
            mlflow.log_metric("test_precision", precision)
            mlflow.log_metric("test_recall", recall)
            mlflow.log_metric("test_f1_score", f1)
            
            # Log classification report
            report_path = "classification_report.txt"
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report)
            mlflow.log_artifact(report_path)
            os.remove(report_path)
            
            # Log confusion matrix
            plt.figure(figsize=(8, 6))
            plt.imshow(cm, interpolation='nearest', cmap='Blues')
            plt.title('Confusion Matrix')
            plt.colorbar()
            
            # Add text annotations
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    plt.text(j, i, str(cm[i, j]), ha='center', va='center', color='black')
            
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            plt.xticks(range(len(class_names)), class_names)
            plt.yticks(range(len(class_names)), class_names)
            plt.tight_layout()
            plt.savefig('confusion_matrix.png')
            mlflow.log_artifact('confusion_matrix.png')
            plt.close()
        
        return report, cm, predictions
    
    def plot_training_history(self, save_plot=False):
        """Plot training history"""
        if self.history is None:
            print("No training history available")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot accuracy
        ax1.plot(self.history.history['accuracy'], label='Training Accuracy')
        ax1.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        
        # Plot loss
        ax2.plot(self.history.history['loss'], label='Training Loss')
        ax2.plot(self.history.history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        
        plt.tight_layout()
        
        if save_plot:
            plt.savefig('training_history.png')
        else:
            plt.show()
        
        plt.close()
    
    def predict_single_image(self, image_path):
        """Predict whether a single image is defective or not"""
        # Load and preprocess image
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.image_size)
        img = img / 255.0
        img = np.expand_dims(img, axis=0)
        
        # Make prediction
        prediction = self.model.predict(img)[0][0]
        
        # Interpret result
        if prediction > 0.5:
            result = "Not Defective"
            confidence = prediction
        else:
            result = "Defective"
            confidence = 1 - prediction
        
        return result, confidence
    
    def save_model(self, filepath):
        """Save the trained model"""
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load a pre-trained model"""
        self.model = keras.models.load_model(filepath)
        print(f"Model loaded from {filepath}")

def create_sample_data_structure():
    """Create sample data directory structure"""
    base_dir = "sample_data"
    
    # Create directories
    os.makedirs(f"{base_dir}/defective", exist_ok=True)
    os.makedirs(f"{base_dir}/not_defective", exist_ok=True)
    
    print(f"Created sample data structure:")
    print(f"  {base_dir}/")
    print(f"    defective/")
    print(f"    not_defective/")
    print(f"\nPlace your training images in these folders:")
    print(f"- Put defective images in '{base_dir}/defective/'")
    print(f"- Put non-defective images in '{base_dir}/not_defective/'")

def main():
    """Main training function"""
    # Create sample data structure
    create_sample_data_structure()
    
    # Check if data exists
    data_dir = "sample_data"
    if not os.path.exists(data_dir):
        print(f"Data directory '{data_dir}' not found. Please create it and add your images.")
        return
    
    # Count images in each class
    defective_count = len(os.listdir(f"{data_dir}/defective")) if os.path.exists(f"{data_dir}/defective") else 0
    not_defective_count = len(os.listdir(f"{data_dir}/not_defective")) if os.path.exists(f"{data_dir}/not_defective") else 0
    
    print(f"Found {defective_count} defective images")
    print(f"Found {not_defective_count} non-defective images")
    
    if defective_count == 0 or not_defective_count == 0:
        print("Please add images to both 'defective' and 'not_defective' folders before training.")
        return
    
    # Initialize classifier
    classifier = DefectClassifier(image_size=(224, 224), batch_size=32)
    
    # Create and compile model
    model = classifier.create_model()
    print("Model architecture:")
    model.summary()
    
    # Train the model
    print("\nStarting training...")
    history = classifier.train(data_dir, epochs=20)
    
    # Plot training history
    classifier.plot_training_history()
    
    # Save the model
    classifier.save_model("defect_classifier_model.h5")
    
    # Example of how to use the trained model
    print("\nTraining completed!")
    print("To use the model for prediction:")
    print("classifier.predict_single_image('path/to/your/image.jpg')")

if __name__ == "__main__":
    main()
