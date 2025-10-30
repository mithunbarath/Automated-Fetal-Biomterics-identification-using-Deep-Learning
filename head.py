import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy

# === U-Net Model for Skull Segmentation ===
def load_unet_model(model_path):
    model = load_model(model_path)
    model.compile(optimizer=Adam(learning_rate=1e-4), loss=BinaryCrossentropy(), metrics=['accuracy'])
    return model

def predict_segmentation(model, image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    original_size = img.shape[:2][::-1]  # (width, height)
    img_resized = cv2.resize(img, (256, 256)) / 255.0
    img_resized = np.expand_dims(np.expand_dims(img_resized, axis=-1), axis=0)

    prediction = model.predict(img_resized)
    raw_pred = prediction[0, :, :, 0]
    raw_pred_normalized = (raw_pred * 255).astype(np.uint8)
    brightened = np.clip(raw_pred_normalized * 2.0, 0, 255).astype(np.uint8)
    cv2.imwrite("brightened_prediction.png", brightened)

    # Resize prediction back to original image size
    pred_mask_resized = cv2.resize((brightened > 0).astype(np.uint8), original_size, interpolation=cv2.INTER_NEAREST)
    return brightened, pred_mask_resized, img  # also return original grayscale image

# === Ellipse Fitting and Head Circumference Estimation ===
def preprocess_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    kernel = np.ones((5, 5), np.uint8)
    closed = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, iterations=2)
    return closed

def extract_all_contours(binary_image):
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def fit_ellipse_to_arcs(contours):
    return [cv2.fitEllipse(c) for c in contours if len(c) >= 5]

def merge_ellipses(ellipses):
    if len(ellipses) < 2:
        return None
    (x1, y1), (w1, h1), a1 = ellipses[0]
    (x2, y2), (w2, h2), a2 = ellipses[1]
    return ((x1 + x2) / 2, (y1 + y2) / 2), ((w1 + w2) / 2, (h1 + h2) / 2), (a1 + a2) / 2

def calculate_ellipse_circumference(ellipse, pixel_size=1.0):
    if ellipse is None:
        return None
    _, (a, b), _ = ellipse
    a /= 2
    b /= 2
    return np.pi * (3*(a+b) - np.sqrt((3*a + b)*(a + 3*b))) * pixel_size

# === Evaluation Metrics ===
def dice_coefficient(mask1, mask2):
    intersection = np.logical_and(mask1, mask2)
    return (2 * intersection.sum()) / (mask1.sum() + mask2.sum())

def intersection_over_union(mask1, mask2):
    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)
    return intersection.sum() / union.sum()

def mean_absolute_error(true, pred): return np.abs(true - pred)
def mean_squared_error(true, pred): return (true - pred) ** 2

# === MAIN WORKFLOW ===

# ==== PATHS ====
model_path = "C:/Users/navee/PycharmProjects/final_head/fetal_head_unet.h5"
test_image_path = r"C:\Users\navee\Downloads\Dataset\HC images 177\training_set\training_set\997_HC.png"
ground_truth_path = r"C:\Users\navee\Downloads\Dataset\HC images 177\training_set\Annotations\997_HC_Annotation.png"

# Load U-Net model
model = load_unet_model(model_path)

# Run segmentation prediction
brightened_img, pred_mask, original_img = predict_segmentation(model, test_image_path)

# Ellipse estimation
binary_image = preprocess_image("brightened_prediction.png")
contours = extract_all_contours(binary_image)
ellipses = fit_ellipse_to_arcs(contours)

# Circumference Estimation
original_measurement_mm = 324
measured_pixels = 1200  # You should replace this with actual measured pixel count
pixel_size = original_measurement_mm / measured_pixels

indiv_circ = [calculate_ellipse_circumference(e, pixel_size) for e in ellipses]
total_indiv_circ = sum(filter(None, indiv_circ))

final_ellipse = merge_ellipses(ellipses)
merged_hc = calculate_ellipse_circumference(final_ellipse, pixel_size)

# Visualization
output = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)
for e in ellipses:
    cv2.ellipse(output, e, (0, 255, 0), 2)
if final_ellipse:
    cv2.ellipse(output, final_ellipse, (0, 0, 255), 2)

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.title("Original Ultrasound")
plt.imshow(original_img, cmap='gray')
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title("Head Circumference")
plt.imshow(output)
if final_ellipse:
    plt.xlabel(f"Estimated HC: {merged_hc:.2f} mm\nTotal from contours: {total_indiv_circ:.2f} mm")
plt.axis("off")

plt.tight_layout()
plt.show()

# Print Results
print(f"Total Predicted Head Circumference: {total_indiv_circ:.2f} mm")
# print(f"Final Estimated HC (Red): {merged_hc:.2f} mm" if merged_hc else "Fit Failed")
bpd=(total_indiv_circ/3.14)/2
ga=(bpd+15)/2
print(f"Total Biparietal Diameter: {bpd:.2f}")
print(f"Gestational Age: {ga:.2f}")
print("\n")
print("Evaluation Metrics:")
# === EVALUATION ===
gt_mask = cv2.imread(ground_truth_path, cv2.IMREAD_GRAYSCALE)
gt_mask = (gt_mask > 0).astype(np.uint8)

# Resize pred_mask to ground truth size if needed
if gt_mask.shape != pred_mask.shape:
    pred_mask = cv2.resize(pred_mask, (gt_mask.shape[1], gt_mask.shape[0]), interpolation=cv2.INTER_NEAREST)

# Evaluation metrics
dice = dice_coefficient(gt_mask, pred_mask)
iou = intersection_over_union(gt_mask, pred_mask)
mae = mean_absolute_error(original_measurement_mm, merged_hc)
mse = mean_squared_error(original_measurement_mm, merged_hc)

print(f"Dice Coefficient: {dice:.4f}")
print(f"IoU: {iou:.4f}")
print(f"MAE: {mae:.2f} mm")
print(f"MSE: {mse:.2f} mm²")
