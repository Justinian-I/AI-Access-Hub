import os
from mon_connect import database  # MongoDB connection
from PIL import Image, ImageOps


def preprocess_image(image_path):
    """
    Preprocessing function to resize and normalize images.
    """
    # Load the image
    image = Image.open(image_path)
    # Resize the image to 256x256
    image = image.resize((256, 256))
    # Convert to grayscale (optional, remove if you want to keep the original color)
    image = ImageOps.grayscale(image)
    # Normalize the image (scale pixel values to range [0, 1])
    image = ImageOps.autocontrast(image)  # Optional: Normalize contrast globally
    # Convert image to a normalized numpy array
    import numpy as np
    image_array = np.asarray(image, dtype=np.float32) / 255.0  # Normalize pixel values
    # Convert back to PIL image for saving
    normalized_image = Image.fromarray((image_array * 255).astype(np.uint8))
    return normalized_image


def preprocess_single_dataset():
    """
    Automatically detects the dataset folder in 'data/', preprocesses its contents,
    saves them in 'preprocessed_data/', and deletes the folder in 'data/'.
    """
    data_dir = r"C:\Users\Mohammad\Graduation_Project\data"  # Input directory
    preprocessed_dir = r"C:\Users\Mohammad\Graduation_Project\pre_data"  # Output directory

    # Get the single dataset folder in 'data/'
    dataset_folders = [f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))]
    if len(dataset_folders) != 1:
        raise ValueError("Expected exactly one folder in 'data/', but found: {}".format(dataset_folders))

    dataset_name = dataset_folders[0]
    input_dir = os.path.join(data_dir, dataset_name)
    output_dir = os.path.join(preprocessed_dir, dataset_name)

    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Process each image
    for image_file in os.listdir(input_dir):
        input_path = os.path.join(input_dir, image_file)
        output_path = os.path.join(output_dir, image_file)

        if os.path.isfile(input_path):  # Skip directories
            processed_image = preprocess_image(input_path)
            processed_image.save(output_path)

    # Save dataset metadata to MongoDB
    dataset_collection = database["datasets"]
    dataset_document = {
        "dataset_name": dataset_name,
        "directory": output_dir
    }
    dataset_collection.insert_one(dataset_document)
    print(f"Dataset '{dataset_name}' processed and added to MongoDB collection.")

    # Delete the input directory
    for file in os.listdir(input_dir):
        os.remove(os.path.join(input_dir, file))
    os.rmdir(input_dir)
    print(f"Deleted dataset folder: {input_dir}")

# Example usage
if __name__ == "__main__":
    preprocess_single_dataset()
