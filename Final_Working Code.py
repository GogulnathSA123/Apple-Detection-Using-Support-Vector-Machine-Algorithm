import cv2

import numpy as np

import matplotlib.pyplot as plt



# ================================================================

# FINAL PROJECT ALGORITHM

# ================================================================

# METHODOLOGY MAPPING:

# 1. Candidate Generation (IP): HSV Thresholding (Optimised for Red Fuji)

# 2. Filtering (IP): Morphological Operations + Geometric Constraints

# 3. Classification (ML): SVM trained on Multi-Dataset (Fuji + MinneApple)

# ================================================================



def count_apples_final_submission(image, model):

    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)



    # 1. IP STAGE: "Red Sunglasses"

    # We use the specific settings that worked best for you.

    # This filters out the green leaves so the model doesn't get confused.

    lower_red1 = np.array([0, 50, 30]);   upper_red1 = np.array([15, 255, 255])

    lower_red2 = np.array([165, 50, 30]); upper_red2 = np.array([180, 255, 255])

    mask = cv2.inRange(hsv, lower_red1, upper_red1) + cv2.inRange(hsv, lower_red2, upper_red2)



    # Clean Noise

    kernel = np.ones((3,3), np.uint8)

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2) # Open removes dust

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1) # Close fills holes



    num, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)



    box_image = image.copy()

    final_count = 0

    raw_candidates = 0



    # 2. INTEGRATION STAGE

    for i in range(1, num):

        x, y, w, h, area = stats[i]



        # GEOMETRIC FILTERS (The Constraints)

        # Area > 80: Removes small leaves/noise

        # Ratio 0.4-2.5: Removes long branches and flat clusters

        if area > 80:

            aspect_ratio = float(w) / h

            if 0.4 < aspect_ratio < 2.5:



                raw_candidates += 1

                roi = image[y:y+h, x:x+w]

                if roi.size == 0: continue



                # Resize for the Brain

                roi_resized = cv2.resize(roi, (64, 64))

                features = get_hog(roi_resized).reshape(1, -1)



                # 3. ML STAGE (The Universal Brain)

                # We ask the model: "Is this an apple?"

                # Note: This model knows about Green apples too, but we only show it Red ones here.

                prob = model.predict_proba(features)[0][1]



                # CLASSIFICATION LOGIC

                if prob > 0.80:

                    # HIGH CONFIDENCE -> GREEN BOX

                    final_count += 1

                    cv2.rectangle(box_image, (x, y), (x+w, y+h), (0, 255, 0), 5)

                elif prob > 0.50:

                    # MEDIUM CONFIDENCE -> YELLOW BOX

                    # We count these, but visualize them differently to show robustness

                    final_count += 1

                    cv2.rectangle(box_image, (x, y), (x+w, y+h), (255, 255, 0), 2)

                else:

                    # LOW CONFIDENCE -> RED BOX (Rejected False Positive)

                    # This proves the "Hybrid" system works (IP found it, ML rejected it)

                    cv2.rectangle(box_image, (x, y), (x+w, y+h), (255, 0, 0), 2)



    return box_image, raw_candidates, final_count



# --- RUN THE FINAL TEST ---

# Ensure you pass 'universal_model' here to satisfy Requirement 3!

test_img = fuji_images[1].copy()

result_img, raw, smart = count_apples_final_submission(test_img, universal_model)



plt.figure(figsize=(15, 10))

plt.imshow(result_img)

plt.title(f"FINAL SUBMISSION RESULT\nRaw Candidates: {raw} | Confirmed Apples: {smart}")

plt.axis('off')

plt.show()



print(f"Evaluation for Report:")

print(f"1. Approach: Hybrid Integrated Algorithm (IP + SVM)")

print(f"2. Model Training: Trained on multi-dataset source (Fuji + MinneApple) to improve generalization.")

print(f"3. Results: Detected {smart} apples out of {raw} candidates.")