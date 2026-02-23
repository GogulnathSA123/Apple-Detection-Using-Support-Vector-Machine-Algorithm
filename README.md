# Apple-Detection-Using-Support-Vector-Machine-Algorithm

# Overview
Apple detection using a Support Vector Machine (SVM) is a supervised machine learning approach used to classify image regions as either apple or non-apple.
This is a binary classification problem in computer vision. The system must identify apples despite challenges such as:
1)Changing lighting conditions
2)Background clutter (leaves, branches)
3)Partial occlusion
4)Variations in apple size and orientation
To solve this efficiently without deep learning, we used:
HOG (Histogram of Oriented Gradients) for feature extraction.
Support Vector Machine (SVM) for classification

The goal is to automatically detect apples in images captured from an orchard or controlled environment. This is useful for:

1)Robotic harvesting
2)Yield estimation
3)Quality inspection
4)Precision agriculture

# Methodology
Image Preprocessing:
    Before extracting features, preprocessing improves consistency and reduces noise.

🔹 Resizing

Images are resized to a fixed dimension (e.g., 64×128 or 128×128).
This ensures:

Uniform feature vector size

Reduced computational complexity

🔹 Grayscale Conversion

HOG primarily relies on gradient intensity rather than color.
Converting to grayscale:

Reduces dimensionality

Focuses on edge structure

Improves speed

🔹 Normalization

Intensity normalization reduces illumination effects.
This ensures that:

Bright sunlight does not dominate gradients

Shadows do not distort detection

Preprocessing improves robustness and stability of detection.

# Feature Extraction Using HOG:
  HOG extracts shape and edge-based features from an image.

Apples are approximately circular objects with strong boundary gradients.
HOG captures this boundary structure effectively.

🔹 Step 1: Gradient Computation

For each pixel:

Compute horizontal gradient (Gx)

Compute vertical gradient (Gy)
This step detects edges and contours.

🔹 Step 2: Cell Division

The image is divided into small regions called cells (e.g., 8×8 pixels).

For each cell:

Compute a histogram of gradient orientations (e.g., 9 bins covering 0–180°).

Each pixel votes into a bin weighted by its gradient magnitude.

This creates a local shape descriptor.

🔹 Step 3: Block Normalization

Cells are grouped into blocks (e.g., 2×2 cells).

Why normalize?

Makes features invariant to lighting changes

Reduces contrast sensitivity

Improves robustness

🔹 Final HOG Feature Vector

All normalized histograms are concatenated into one long vector.

This vector represents:

Shape

Edge direction

Local structure

This becomes the input to the SVM classifier.

# Support Vector Machine:
  SVM is a supervised learning algorithm used for classification.

🔹 Training Phase

We provide labeled data:

HOG features of apple images → label = 1

HOG features of non-apple images → label = 0

SVM finds an optimal hyperplane:



That separates the two classes with maximum margin.

# Detection Phase:
 For a new image:

🔹 Step 1: Sliding Window

Move a fixed-size window across the image.

🔹 Step 2: Extract HOG Features

Compute HOG for each window region.

🔹 Step 3: SVM Prediction

Feed feature vector into trained SVM.

→ Apple detected

Otherwise:
→ Non-apple

🔹 Step 4: Bounding Box Output

Mark detected regions with bounding boxes.

Non-Maximum Suppression (NMS) to remove duplicate detections.

<img width="1053" height="744" alt="Screenshot 2026-02-23 204614" src="https://github.com/user-attachments/assets/73ef524a-b9a9-4e24-8d20-9036f5759538" />




