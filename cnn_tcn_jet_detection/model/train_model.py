import pathlib
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras import layers, models, losses, optimizers
from tcn import TCN  
import json
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ReduceLROnPlateau
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.manifold import TSNE
import seaborn as sns

# Data loading and preprocessing
FILE_DIR = "DATASET_PATH"
DATA_PATH = pathlib.Path(FILE_DIR)
IMG_SIZE = (128, 128)
BATCH_SIZE = 16  
EPOCHS = 1

datagen = image.ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.3
)

train_generator = datagen.flow_from_directory(
    FILE_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='sparse', subset='training', shuffle=True
)
val_generator = datagen.flow_from_directory(
    FILE_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='sparse', subset='validation', shuffle=False
)
nClass = len(train_generator.class_indices)

# Model definition
model = models.Sequential([
    layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
    layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    
    layers.Reshape((-1, 256)),  
    TCN(512, return_sequences=True),
    TCN(512, return_sequences=True),
    TCN(256, return_sequences=True),
    TCN(256, return_sequences=True),
    TCN(128, return_sequences=True),
    TCN(128, return_sequences=False),
    layers.Dense(256, activation='relu', name="dense_features"),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(nClass, activation='softmax')
])

# Compile model
opt = tf.keras.optimizers.Adam(learning_rate=1e-3)
model.compile(optimizer=opt, loss=losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])

lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)

model.summary()

# Train the model
history = model.fit(train_generator, validation_data=val_generator, epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=[lr_scheduler])

# Save model and class indices
model.save("MODEL_FILE_PATH")
with open("JSON_FILE_PATH", 'w') as f:
    json.dump(train_generator.class_indices, f)
print("Model and class indices saved successfully.")

# Plot training history
def plot_training_history(history):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend() 
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.show()

plot_training_history(history)

# Evaluate and predict
y_true = np.concatenate([val_generator[i][1] for i in range(len(val_generator))])
y_pred_probs = model.predict(val_generator)
y_pred = np.argmax(y_pred_probs, axis=1)

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=val_generator.class_indices, yticklabels=val_generator.class_indices)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Evaluate
test_loss, test_accuracy = model.evaluate(val_generator)
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
