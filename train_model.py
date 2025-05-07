import autokeras as ak
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
import os
import datetime
from sklearn.model_selection import train_test_split
from PIL import Image
import keras_tuner
import argparse
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping, CSVLogger
from tensorflow.keras.preprocessing import image
import json 
import random
# In[2]:
# Argument parser
parser = argparse.ArgumentParser(description="Train a model with specified classes.")
#parser.add_argument("--query", type=str, help="Search query for new images", required=False)
parser.add_argument("--classes", type=str, help="Comma-separated list of class names", required=True)
parser.add_argument("--user", type=str, help="User name for saving the model", required=True)  # New argument

args = parser.parse_args()
#query = args.query
if args.classes:
    wanted_classes = args.classes.split(",")
else:
    wanted_classes = []
user = args.user

# Training logic remains the same; adjust `wanted_classes` dynamically
print("Training model with classes:", wanted_classes)
print(f"user: {args.user}")


# Define the path to the image files
dataset_path = r'C:\Users\Mohammad\Graduation_Project\pre_data'  
###############
#prepare dataset for boolean classification
def prepare_dataset_Bool(base_dir, positive_folder_name):
    file_paths = []
    labels = []
    class_to_label = {1: positive_folder_name, 0: "others"}

    positive_folder = os.path.join(base_dir, positive_folder_name)

    # Check if the positive folder exists
    if not os.path.exists(positive_folder):
        raise ValueError(f"Folder {positive_folder_name} does not exist in {base_dir}.")

    # Get all files in the positive folder and assign label 1
    positive_files = [os.path.join(positive_folder, fname) for fname in os.listdir(positive_folder)]
    file_paths.extend(positive_files)
    labels.extend([1] * len(positive_files))
    print(f"Number of positive images (label 1) from '{positive_folder_name}': {len(positive_files)}")

    # Sample from other folders
    other_folders = [f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f)) and f != positive_folder_name]
    all_other_files = []

    for folder in other_folders:
        folder_path = os.path.join(base_dir, folder)
        files_in_folder = [os.path.join(folder_path, fname) for fname in os.listdir(folder_path)]
        all_other_files.extend(files_in_folder)

    print(f"Total number of images in other folders: {len(all_other_files)}")
    print(f"Number of other folders: {len(other_folders)}")

    # Randomly sample images from other folders
    num_samples = len(positive_files)
    sampled_files = random.sample(all_other_files, min(num_samples, len(all_other_files)))

    # Assign label 0 to the sampled files
    file_paths.extend(sampled_files)
    labels.extend([0] * len(sampled_files))
    print(f"Number of sampled images (label 0) from 'others': {len(sampled_files)}")

    # Summary of dataset
    print(f"Total number of images: {len(file_paths)}")
    print(f"Label distribution: 1 -> {labels.count(1)}, 0 -> {labels.count(0)}")

    return file_paths, labels, class_to_label

    

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
selected_class = wanted_classes[0]
# Determine mode based on the number of total classes
if (len(wanted_classes) == 1):
    boolean_mode = True
    
    print(f"Boolean classification mode selected for class: {selected_class}")
else:
    boolean_mode = False
    print("Multi-class classification mode selected.")

# Gather file paths and labels
file_paths = []
labels = []

if boolean_mode:

     file_paths, labels, class_to_label = prepare_dataset_Bool(dataset_path, selected_class)
     print("\nSample of file paths and labels:")
     for i in range(min(5, len(file_paths))):  # Print up to 5 samples
         print(f"{file_paths[i]} -> Label: {labels[i]}")
     start_index = 7000
     end_index = min(start_index + 5, len(file_paths))  # Ensure we don't go beyond the length of the list

     print(f"\nDisplaying file paths and labels starting from index {start_index}:")

     for i in range(start_index, end_index):
         print(f"{file_paths[i]} -> Label: {labels[i]}")

        # Print label distribution
     print(f"\nTotal number of images: {len(file_paths)}")
     print(f"Label distribution: 1 -> {labels.count(1)}, 0 -> {labels.count(0)}")

        # Print label to class mapping
     print("\nLabel to class mapping:")
     for label, class_name in class_to_label.items():
         print(f"Label {label}: {class_name}")
else:    
    class_to_label = {class_name: idx for idx, class_name in enumerate(wanted_classes)}
    
    for class_name, label in class_to_label.items():
        class_dir = os.path.join(dataset_path, class_name)
        if os.path.exists(class_dir):
            for fname in os.listdir(class_dir):
                file_paths.append(os.path.join(class_dir, fname))
                labels.append(label)
                
    print("\nSample of file paths and labels:")
    for i in range(min(5, len(file_paths))):  # Print up to 5 samples
        print(f"{file_paths[i]} -> Label: {labels[i]}")

        # Print label distribution
    print(f"\nTotal number of images: {len(file_paths)}")
    print(f"Label distribution: 1 -> {labels.count(1)}, 0 -> {labels.count(0)}")

        # Print label to class mapping
    print("\nLabel to class mapping:")
    for label, class_name in class_to_label.items():
        print(f"Label {label}: {class_name}")

# Split into training and validation subsets
train_paths, val_paths, train_labels, val_labels = train_test_split(
    file_paths, labels, test_size=0.2, random_state=123
)

# Create a TensorFlow Dataset
def preprocess_image(filename, label):
    image = tf.io.read_file(filename)
    image = tf.image.decode_image(image, channels=1)  # Grayscale
    image.set_shape([None, None, 1])  # Explicitly set shape (height, width, channels)
    image = tf.image.resize(image, [64, 64])  # Resize to the target size
    image = tf.cast(image, tf.float32) / 255.0  # Normalize to [0, 1]
    return image, label

train_data = tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
train_data = train_data.map(preprocess_image).batch(32)

val_data = tf.data.Dataset.from_tensor_slices((val_paths, val_labels))
val_data = val_data.map(preprocess_image).batch(32)

# Print the class names
print("Loaded classes:", wanted_classes)

# # 2. Convert the dataset to NumPy arrays
def dataset_to_numpy(dataset):
    images = []
    labels = []
    for img, label in dataset:
        images.append(img.numpy())  # Convert tensor to numpy array
        labels.append(label.numpy())  # Convert tensor to numpy array
#         print("Processing image...")  # Print a message for each image
    return np.concatenate(images), np.concatenate(labels)

# Convert training and validation data
X_train, y_train = dataset_to_numpy(train_data)
X_val, y_val = dataset_to_numpy(val_data)


print(X_train.shape)
print(X_val.shape)


# Create an ImageClassifier object
clf = ak.ImageClassifier(
    overwrite=True,
    max_trials=1,
    max_model_size=53567364
)

# Define the EarlyStopping callback
early_stopping = EarlyStopping(
    monitor='val_loss',       
    patience=5,                
    restore_best_weights=True  
)

# Define the CSVLogger callback
csv_logger = CSVLogger(
    "training_log.csv",  # File to save the logs
    append=False         # Overwrite the file each time
)

# Train the model using both callbacks
history = clf.fit(
    X_train,                      # Training dataset
    y_train,                      # Training labels
    validation_data=(X_val, y_val),  # Validation dataset and labels
    epochs=30,                   # Total epochs to train
    callbacks=[early_stopping, csv_logger]  # Include both callbacks
)


val_loss, val_accuracy = clf.evaluate(X_val, y_val)
print(f"Validation Loss: {val_loss:.4f}")
print(f"Validation Accuracy: {val_accuracy:.4f}")

# Retrieve the best model from AutoKeras after training
best_model = clf.export_model()
# Display the summary of the best model
best_model.summary()

def create_model_directory(user, model_name):
    base_dir = "models"
    user_dir = os.path.join(base_dir, user)
    
    # Ensure model_name is a valid string
    if model_name is None:
        model_name = "default_model"  # Or any other fallback name
    
    model_dir = os.path.join(user_dir, model_name)

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    return model_dir

# Export the best model found by AutoKeras
model = clf.export_model()

model_name = f"{'_'.join(wanted_classes)}"
models_base_path = r"C:\Users\Mohammad\Graduation_Project\models"
model_directory = create_model_directory(user, model_name)
print(f"Model name: {model_name}")
print(f"Model_directory: {model_directory}")

print(f"selected class: {selected_class}")



# Ensure the directory exists
os.makedirs(model_directory, exist_ok=True)

# Save the model as a .h5 file in the dynamic directory
model_file_path = os.path.join(model_directory, f"{model_name}.keras")
model.save(model_file_path, save_format="keras")
print(f"Model successfully saved to: {model_file_path}")

# Save metrics to JSON
metrics = {"val_loss": val_loss, "val_accuracy": val_accuracy}
metrics_path = os.path.join(model_directory, "metrics.json")
with open(metrics_path, "w") as f:
    json.dump(metrics, f)

print(f"Metrics saved to: {metrics_path}")


def preprocess_image(img_path, target_size=(64, 64)):
    """Load and preprocess a single image for prediction."""
    img = image.load_img(img_path, target_size=target_size, color_mode="grayscale")
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize the image
    return img_array


def predict_folder_with_accuracy(model, folder_path, true_label):
    """Predict labels for all images in a folder and calculate accuracy based on true label."""
    predictions = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            predicted_label = predict_single_image(model, file_path)
            predictions.append(predicted_label)

    # Calculate accuracy
    correct_predictions = sum([1 for pred in predictions if pred == true_label])
    accuracy = correct_predictions / len(predictions) * 100
    print(f"Accuracy for folder '{true_label}': {accuracy:.2f}%")
    return accuracy