import os
import numpy as np
import pandas as pd
import zipfile
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

# Step 1: Extract the training dataset
zip_file_path = '/content/archive.zip'
if os.path.exists(zip_file_path):
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall('/content/training_data')

# Training Data Paths
data_folder = '/content/training_data/data'
csv_file = os.path.join(data_folder, 'output.csv')
frames_folder = os.path.join(data_folder, 'frames')

# Step 2: Read the CSV file
data = pd.read_csv(csv_file)
print(f"Number of images in dataset: {len(data)}")
print(data.head())

# Step 3: Preprocess images
def preprocess_images(image_folder, image_paths, target_size=(128, 128)):
    """
    Preprocesses images by loading, resizing, and normalizing.
    
    Args:
        image_folder (str): Base folder where images are stored.
        image_paths (list): List of image paths relative to the base folder.
        target_size (tuple): Target size for image resizing.
        
    Returns:
        numpy.ndarray: Array of preprocessed image data.
    """
    images = []
    for img_path in image_paths:
        # Extract the filename only (e.g., seq_00001.jpg)
        img_filename = os.path.basename(img_path)
        full_path = os.path.join(image_folder, img_filename)  # Correct path
        try:
            # Load and preprocess the image
            img = load_img(full_path, target_size=target_size)
            img_array = img_to_array(img) / 255.0  # Normalize
            images.append(img_array)
        except FileNotFoundError:
            print(f"Warning: File not found - {full_path}")
    return np.array(images)

# Extract image paths and counts
image_paths = data['image']  # Use the 'image' column directly
counts = data['count'].values

# Preprocess images
images = preprocess_images(frames_folder, image_paths)

# Split into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(images, counts, test_size=0.2, random_state=42)
print(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")

# Step 4: Build the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D((2, 2)),
    Dropout(0.2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='linear')  # Regression output
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error', metrics=['mae'])

# Step 5: Train the model
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50, batch_size=32, callbacks=[early_stopping])

# Visualize the training process
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Training and Validation Loss')
plt.show()

# Step 6: Extract the test dataset
test_zip_file = '/content/image sets.zip'
if os.path.exists(test_zip_file):
    with zipfile.ZipFile(test_zip_file, 'r') as zip_ref:
        zip_ref.extractall('/content/testing_data')

test_folder = '/content/testing_data'

# Step 7: Load and preprocess test images
test_images = [f for f in os.listdir(test_folder) if f.endswith('.jpg') or f.endswith('.png')]
test_images_array = preprocess_images(test_folder, test_images)

# Step 8: Make predictions on the test images
predictions = model.predict(test_images_array)

# Display predictions
print("Test Predictions:")
for i, img_name in enumerate(test_images):
    print(f"Image: {img_name}, Predicted Count: {predictions[i][0]:.2f}")
