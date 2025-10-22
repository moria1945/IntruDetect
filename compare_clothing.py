import torch
import torch.nn as nn
from torchvision import models, transforms
import cv2
import numpy as np
import os
from sklearn.cluster import MiniBatchKMeans
from collections import Counter

# --- 1. הגדרות וטעינת מודלים מאומנים ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# טרנספורמציות זהות לאלה שבאימון!
data_transform_eval = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def load_clothing_model(model_path, num_classes, device):
    """Loads a pre-trained ResNet18 model and then loads custom weights."""
    model = models.resnet18(pretrained=False) # We load custom weights, so no need for ImageNet pretrained here
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}. Please ensure the model was trained and saved.")

    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval() # Set model to evaluation mode
    return model

# Load coat model
COAT_MODEL_PATH = './models/best_coat_model.pth'
coat_model_classes = ['with_coat', 'without_coat']
try:
    coat_model = load_clothing_model(COAT_MODEL_PATH, len(coat_model_classes), DEVICE)
    print("Coat model loaded successfully.")
except FileNotFoundError as e:
    print(f"Error loading coat model: {e}. Coat detection will not be available.")
    coat_model = None

# Load white shirt model
SHIRT_MODEL_PATH = './models/best_white_shirt_model.pth'
shirt_model_classes = ['non_white_shirt', 'white_shirt'] # Ensure this order matches your training dataset's class_names
try:
    shirt_model = load_clothing_model(SHIRT_MODEL_PATH, len(shirt_model_classes), DEVICE)
    print("White shirt model loaded successfully.")
except FileNotFoundError as e:
    print(f"Error loading white shirt model: {e}. White shirt detection will not be available.")
    shirt_model = None


# --- 2. Detection and Classification Functions ---

def predict_clothing(image_crop_path, model, class_names, transform, device):
    """Performs classification on a given image crop."""
    if model is None:
        return "Model not loaded"

    img = cv2.imread(image_crop_path)
    if img is None:
        print(f"Warning: Could not load image from {image_crop_path}. Skipping prediction.")
        return "N/A"

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_tensor = transform(img_rgb).unsqueeze(0) # Add batch dimension
    img_tensor = img_tensor.to(device)

    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted = torch.max(outputs, 1)
        return class_names[predicted.item()]

def get_dominant_color_from_shirt_area(image_crop_path):
    """
    Attempts to extract a dominant color from an estimated shirt area.
    This is a basic heuristic; a segmentation model would be more accurate.
    """
    img = cv2.imread(image_crop_path)
    if img is None:
        print(f"Warning: Could not load image from {image_crop_path} for color analysis. Skipping.")
        return "N/A"

    h, w, _ = img.shape
    # Estimate shirt area (adjust these values based on your typical crop content)
    # This takes the middle horizontal section of the crop, and 40% height of the image.
    shirt_area = img[int(h*0.2):int(h*0.6), int(w*0.2):int(w*0.8)]

    if shirt_area.size == 0 or shirt_area.shape[0] == 0 or shirt_area.shape[1] == 0:
        return "N/A" # Empty or invalid region

    # Reshape pixels for K-Means
    pixels = shirt_area.reshape(-1, 3)
    
    try:
        # Use MiniBatchKMeans for faster processing on potentially large pixel sets
        kmeans = MiniBatchKMeans(n_clusters=3, n_init=10, random_state=0, n_jobs=-1).fit(pixels)
        dominant_color_bgr = kmeans.cluster_centers_[np.argmax(np.bincount(kmeans.labels_))]
    except ValueError: # Occurs if not enough pixels to form clusters
        return "N/A"

    dominant_color_rgb = dominant_color_bgr[::-1] # Convert BGR to RGB

    # Simple threshold for white color
    # RGB values > 200 generally indicate bright colors, often white or near-white
    if all(c > 200 for c in dominant_color_rgb):
        return "white"
    elif all(c < 50 for c in dominant_color_rgb): # Basic threshold for black
        return "black"
    else:
        # You can add more sophisticated color categorization here if needed
        return "other_color"

# --- 3. Main Comparison Logic ---

def analyze_group_clothing(person_crop_paths):
    all_people_data = []

    for i, path in enumerate(person_crop_paths):
        person_data = {'id': f'person_{i}', 'path': path}
        
        # Analyze coat
        if coat_model:
            coat_pred = predict_clothing(path, coat_model, coat_model_classes, data_transform_eval, DEVICE)
            person_data['coat'] = coat_pred
        else:
            person_data['coat'] = "N/A (Model not loaded)"

        # Analyze shirt color
        if shirt_model:
            shirt_pred = predict_clothing(path, shirt_model, shirt_model_classes, data_transform_eval, DEVICE)
            person_data['shirt_white_classification'] = shirt_pred
            
            # Also get a general dominant color
            dominant_color = get_dominant_color_from_shirt_area(path)
            person_data['shirt_general_color'] = dominant_color
        else:
            person_data['shirt_white_classification'] = "N/A (Model not loaded)"
            person_data['shirt_general_color'] = "N/A"

        all_people_data.append(person_data)

    print("\n--- Clothing Analysis Per Person ---")
    for person_data in all_people_data:
        print(f"{person_data['id']}: Coat: {person_data['coat']}, Shirt (classified): {person_data['shirt_white_classification']}, General Color: {person_data['shirt_general_color']}")

    anomalies = []

    # Check for coat anomalies
    if coat_model and len(all_people_data) > 1:
        with_coat_count = sum(1 for p in all_people_data if p['coat'] == 'with_coat')
        
        if with_coat_count == 1: # Only one person wears a coat
            for p_data in all_people_data:
                if p_data['coat'] == 'with_coat':
                    anomalies.append(f"{p_data['id']} is anomalous: wears a coat while others don't.")
        elif with_coat_count == len(all_people_data) - 1: # Everyone but one wears a coat
            for p_data in all_people_data:
                if p_data['coat'] == 'without_coat':
                    anomalies.append(f"{p_data['id']} is anomalous: doesn't wear a coat while others do.")

    # Check for shirt color anomalies
    if shirt_model and len(all_people_data) > 1:
        # Anomaly based on 'white_shirt' classification
        white_shirt_count = sum(1 for p in all_people_data if p['shirt_white_classification'] == 'white_shirt')
        
        if white_shirt_count == 1:
             for p_data in all_people_data:
                if p_data['shirt_white_classification'] == 'white_shirt':
                    anomalies.append(f"{p_data['id']} is anomalous: wears a white shirt while others don't.")
        elif white_shirt_count == len(all_people_data) - 1:
            for p_data in all_people_data:
                if p_data['shirt_white_classification'] == 'non_white_shirt':
                    anomalies.append(f"{p_data['id']} is anomalous: wears a non-white shirt while others wear white.")
        
        # Anomaly based on general dominant color (more nuanced)
        general_colors = [p['shirt_general_color'] for p in all_people_data if p['shirt_general_color'] != 'N/A']
        if general_colors:
            color_counts = Counter(general_colors)
            if len(color_counts) > 1: # If there's more than one color identified
                most_common_color, _ = color_counts.most_common(1)[0]
                
                # If a large majority shares a color, and one person is different
                # Using a threshold, e.g., 80% of the group shares the same color
                if color_counts[most_common_color] / len(general_colors) >= 0.8:
                    for p_data in all_people_data:
                        if p_data['shirt_general_color'] != most_common_color and p_data['shirt_general_color'] != 'N/A':
                             anomalies.append(f"{p_data['id']} is anomalous: shirt color significantly differs from the majority ({p_data['shirt_general_color']} vs. {most_common_color}).")

    if not anomalies:
        print("\nNo significant clothing anomalies found in the group.")
    else:
        print("\n--- Detected Clothing Anomalies ---")
        for anomaly in anomalies:
            print(anomaly)

# --- Example Usage ---
if __name__ == '__main__':
    # This is a placeholder list of paths to your cropped images.
    # You need to ensure these files exist in your 'temp_data_for_cpp/' directory.
    # The 'main' part of your project (e.g., Cpp_m.exe and its Python scripts)
    # should generate these focused_person_crop_X.jpg files.
    sample_person_crops = [
        './temp_data_for_cpp/focused_person_crop_0.jpg',
        './temp_data_for_cpp/focused_person_crop_1.jpg',
        './temp_data_for_cpp/focused_person_crop_2.jpg',
        # Add all relevant crop paths here for the group you want to analyze
    ]
    
    # Ensure that 'temp_data_for_cpp/' and 'models/' directories exist
    # and that your trained models are saved in 'models/'.
    analyze_group_clothing(sample_person_crops)