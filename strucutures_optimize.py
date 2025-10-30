import numpy as np
import torch
import torchvision
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt


# -------------------------------
# Model Definition and Loading
# -------------------------------
def get_model():
    # Create a MaskRCNN model with a ResNet50-FPN backbone.
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # Replace the box and mask predictors to match 5 classes (background + 4 structures)
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features,
                                                                                               num_classes=5)
    model.roi_heads.mask_predictor = torchvision.models.detection.mask_rcnn.MaskRCNNPredictor(256, 256, num_classes=5)
    return model


# Set device and load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = get_model().to(device)

try:
    state_dict = torch.load("C:/Users/navee/PycharmProjects/final_abdomen/fetal_abdomen_mask_rcnn_hrnet.pth",
                            map_location=device)
    model.load_state_dict(state_dict)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

model.eval()

# -------------------------------
# Transformation (as used during training)
# -------------------------------
transform = A.Compose([
    A.Resize(512, 512),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

# -------------------------------
# Visualization Settings
# -------------------------------
CLASS_COLORS = {
    1: (255, 0, 0),  # artery - red
    2: (0, 255, 0),  # liver - green
    3: (0, 0, 255),  # stomach - blue
    4: (255, 0, 255)  # vein - magenta
}

CLASS_NAMES = {
    1: "artery",
    2: "liver",
    3: "stomach",
    4: "vein"
}


# -------------------------------
# Debug function for raw predictions
# -------------------------------
def debug_predictions(pred):
    """Print detailed prediction information"""
    print("\nPrediction Debug Info:")
    print(f"Found {len(pred['labels'])} detections")
    if len(pred['labels']) > 0:
        scores = [f"{s:.2f}" for s in pred['scores'].cpu().numpy()]
        classes = [CLASS_NAMES.get(l.item(), 'unknown') for l in pred['labels']]
        print("Score distribution:", scores)
        print("Class distribution:", classes)


# -------------------------------
# Standardize Predictions Function
# -------------------------------
def standardize_predictions(pred, confidence_thresh=0.2):
    """
    For each class (artery, liver, stomach, vein), select the detection with
    the highest confidence above the given threshold. If none exist, return None.
    """
    standardized = {}
    for label in CLASS_NAMES.keys():
        # Get indices for detections of this class with sufficient confidence
        indices = [i for i, l in enumerate(pred['labels'])
                   if l.item() == label and pred['scores'][i].item() >= confidence_thresh]
        if len(indices) == 0:
            standardized[CLASS_NAMES[label]] = None
        else:
            # Select the detection with the highest score
            best_index = max(indices, key=lambda i: pred['scores'][i].item())
            standardized[CLASS_NAMES[label]] = {
                'box': pred['boxes'][best_index].cpu().numpy().astype(int),
                'score': pred['scores'][best_index].item(),
                'mask': pred['masks'][best_index][0].cpu().numpy() if pred.get('masks', None) is not None and pred[
                    'masks'].numel() > 0 else None
            }
    return standardized


# -------------------------------
# Visualization for Standardized Predictions
# -------------------------------
def visualize_standardized(image_tensor, standardized_preds):
    """
    Visualize standardized predictions on the transformed image.
    image_tensor: the transformed image tensor (C,H,W)
    standardized_preds: dict mapping structure names to prediction info or None.
    """
    # Convert tensor back to image (undo normalization)
    image = image_tensor.permute(1, 2, 0).cpu().numpy()
    # Undo normalization (assuming the same mean and std as in training)
    image = (image * np.array([0.229, 0.224, 0.225])) + np.array([0.485, 0.456, 0.406])
    image = np.clip(image, 0, 1)
    image = (image * 255).astype(np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Create a copy for overlaying predictions
    overlay = image.copy()

    for structure, pred_info in standardized_preds.items():
        if pred_info is None:
            continue
        box = pred_info['box']
        score = pred_info['score']
        mask = pred_info['mask']
        # Find the label key from CLASS_NAMES
        label_key = [k for k, v in CLASS_NAMES.items() if v == structure][0]
        color = CLASS_COLORS.get(label_key, (255, 255, 255))

        # Draw bounding box and label
        cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), color, 2)
        cv2.putText(image, f"{structure} {score:.2f}", (box[0], box[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Overlay mask if available
        if mask is not None:
            # Create a binary mask
            mask_bin = (mask > 0.5).astype(np.uint8)
            # Color the overlay where the mask is present
            overlay[mask_bin == 1] = color

    # Blend the original image with the overlay
    result = cv2.addWeighted(image, 0.7, overlay, 0.3, 0)

    # Display the result
    cv2.imshow("Standardized Detection Results", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # Save the result if desired
    cv2.imwrite("standardized_detection_result.jpg", result)


# -------------------------------
# Main Prediction Function
# -------------------------------
def predict(image_path, confidence_thresh=0.2):
    # Load image using OpenCV
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image at {image_path}")
        return
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Transform the image (resize, normalize, and convert to tensor)
    transformed = transform(image=image)
    img_tensor = transformed["image"].unsqueeze(0).to(device)

    # Predict using the model
    with torch.no_grad():
        predictions = model(img_tensor)

    # Debug raw predictions
    debug_predictions(predictions[0])

    # Standardize predictions by selecting one best detection per structure
    standardized_preds = standardize_predictions(predictions[0], confidence_thresh)
    print("\nStandardized Predictions:")
    for k, v in standardized_preds.items():
        print(f"{k}: {v}")

    # Visualize the standardized predictions using OpenCV
    visualize_standardized(transformed["image"], standardized_preds)


# -------------------------------
# Run the Prediction on a Test Image
# -------------------------------
test_image_path = "C:/Users/navee/Downloads/Dataset/Fetal Abdominal Structures/IMAGES/P01_IMG2.png"
predict(test_image_path, confidence_thresh=0.2)
