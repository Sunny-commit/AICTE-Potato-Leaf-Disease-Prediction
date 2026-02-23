# 🍃 Potato Leaf Disease Prediction - CNN Computer Vision

A **deep learning project using Convolutional Neural Networks** to classify and predict potato leaf diseases from image data, enabling farmers to detect crop infections early and prevent yield loss.

## 🎯 Overview

This project provides:
- ✅ Image classification using CNN
- ✅ Transfer learning with pre-trained models
- ✅ Image augmentation techniques
- ✅ Model interpretability (feature maps)
- ✅ Real-time disease detection
- ✅ Deployment-ready pipeline

## 📸 Dataset

```
Potato Leaf Dataset:
├── Classes: 3 (Healthy, Early Blight, Late Blight)
├── Images: ~2000+ samples
├── Resolution: 224x224 RGB
├── Data split: 70% train, 15% val, 15% test
├── Imbalance handling: Class weighting
└── Preprocessing: Normalization, augmentation
```

## 🏗️ CNN Architecture

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

class PotatoDiseaseCNN:
    """CNN for potato disease classification"""
    
    def __init__(self, input_shape=(224, 224, 3), num_classes=3):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = self._build_model()
    
    def _build_model(self):
        """Build custom CNN"""
        model = keras.Sequential([
            # Block 1
            layers.Conv2D(32, (3, 3), activation='relu', 
                         padding='same', input_shape=self.input_shape),
            layers.BatchNormalization(),
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Block 2
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Block 3
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Block 4
            layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # FC layers
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
```

## 🔄 Transfer Learning

```python
class PotatoDiseaseTransferLearning:
    """Transfer learning with pre-trained models"""
    
    def __init__(self, model_name='ResNet50', num_classes=3):
        self.model_name = model_name
        self.num_classes = num_classes
        self.model = self._build_transfer_model()
    
    def _build_transfer_model(self):
        """Build transfer learning model"""
        # Load pre-trained model
        if self.model_name == 'ResNet50':
            base_model = keras.applications.ResNet50(
                input_shape=(224, 224, 3),
                include_top=False,
                weights='imagenet'
            )
        elif self.model_name == 'MobileNetV2':
            base_model = keras.applications.MobileNetV2(
                input_shape=(224, 224, 3),
                include_top=False,
                weights='imagenet'
            )
        elif self.model_name == 'EfficientNetB0':
            base_model = keras.applications.EfficientNetB0(
                input_shape=(224, 224, 3),
                include_top=False,
                weights='imagenet'
            )
        
        # Freeze base model
        base_model.trainable = False
        
        # Add custom layers
        model = keras.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy', keras.metrics.AUC()]
        )
        
        return model
    
    def unfreeze_and_finetune(self, num_layers_to_unfreeze=50):
        """Fine-tune specific layers"""
        base_model = self.model.layers[0]
        
        # Unfreeze last N layers
        for layer in base_model.layers[-num_layers_to_unfreeze:]:
            layer.trainable = True
        
        # Compile with lower learning rate
        self.model.compile(
            optimizer=keras.optimizers.Adam(0.0001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
```

## 🖼️ Image Preprocessing & Augmentation

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2

class PotatoImagePreprocessor:
    """Image preprocessing and augmentation"""
    
    @staticmethod
    def load_and_preprocess(image_path, target_size=(224, 224)):
        """Load and preprocess single image"""
        # Read image
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize
        img = cv2.resize(img, target_size)
        
        # Normalize
        img = img / 255.0
        
        return img
    
    @staticmethod
    def create_augmentation_generator():
        """Create data augmentation"""
        return ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            vertical_flip=True,
            zoom_range=0.2,
            fill_mode='nearest',
            brightness_range=[0.8, 1.2],
            rescale=1./255
        )
    
    @staticmethod
    def create_generators(train_dir, val_dir, batch_size=32):
        """Create train and validation generators"""
        train_datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            zoom_range=0.2,
            rescale=1./255
        )
        
        val_datagen = ImageDataGenerator(rescale=1./255)
        
        train_gen = train_datagen.flow_from_directory(
            train_dir,
            target_size=(224, 224),
            batch_size=batch_size,
            class_mode='categorical'
        )
        
        val_gen = val_datagen.flow_from_directory(
            val_dir,
            target_size=(224, 224),
            batch_size=batch_size,
            class_mode='categorical'
        )
        
        return train_gen, val_gen
```

## 🎨 Model Interpretability

```python
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image as keras_image

class PotatoDiseaseInterpretability:
    """Understand model decisions"""
    
    @staticmethod
    def visualize_feature_maps(model, image, layer_index):
        """Visualize intermediate layer activations"""
        intermediate_model = keras.Model(
            inputs=model.input,
            outputs=model.layers[layer_index].output
        )
        intermediate_predictions = intermediate_model.predict(image)
        
        # Plot feature maps
        img_count = intermediate_predictions.shape[-1]
        fig, axes = plt.subplots(4, 8, figsize=(15, 8))
        axes = axes.flatten()
        
        for idx in range(min(32, img_count)):
            axes[idx].imshow(intermediate_predictions[0, :, :, idx], cmap='viridis')
            axes[idx].set_title(f'Feature {idx}')
            axes[idx].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def grad_cam(model, image, class_index, last_conv_layer_name):
        """Gradient-weighted Class Activation Mapping"""
        grad_model = keras.models.Model(
            [model.inputs],
            [model.get_layer(last_conv_layer_name).output, model.output]
        )
        
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(image)
            loss = predictions[:, class_index]
        
        output = conv_outputs[0]
        grads = tape.gradient(loss, conv_outputs)[0]
        
        gate_f = tf.reduce_mean(grads, axis=(0, 1))
        cam = output @ gate_f[..., tf.newaxis]
        cam = tf.squeeze(cam)
        
        return cam
```

## 📊 Training & Evaluation

```python
class PotatoDiseaseTrainer:
    """Train and evaluate model"""
    
    def __init__(self, model):
        self.model = model
        self.history = None
    
    def train(self, train_gen, val_gen, epochs=50):
        """Train model"""
        callbacks = [
            keras.callbacks.EarlyStopping(monitor='val_loss', patience=5),
            keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3),
            keras.callbacks.ModelCheckpoint(
                'best_model.h5',
                monitor='val_accuracy',
                save_best_only=True
            )
        ]
        
        self.history = self.model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=epochs,
            callbacks=callbacks
        )
        
        return self.history
    
    def plot_training_history(self):
        """Plot training curves"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Accuracy
        ax1.plot(self.history.history['accuracy'], label='Train')
        ax1.plot(self.history.history['val_accuracy'], label='Val')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid()
        
        # Loss
        ax2.plot(self.history.history['loss'], label='Train')
        ax2.plot(self.history.history['val_loss'], label='Val')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid()
        
        plt.tight_layout()
        plt.show()
```

## 💡 Interview Talking Points

**Q: Why CNN for image classification?**
```
Answer:
- Spatial hierarchy: Learns local patterns
- Parameter sharing: Reduces parameters
- Translation invariance: Robust to shifts
- Proven effective on vision tasks
```

**Q: Transfer learning benefits?**
```
Answer:
- Pre-trained weights capture general features
- Faster convergence
- Better performance on small datasets
- Reduced computational cost
```

## 🌟 Portfolio Value

✅ CNN architecture design
✅ Transfer learning
✅ Image preprocessing & augmentation
✅ Model interpretability (Grad-CAM)
✅ Disease classification
✅ Agricultural ML application

---

**Technologies**: TensorFlow, Keras, OpenCV, NumPy, Matplotlib

