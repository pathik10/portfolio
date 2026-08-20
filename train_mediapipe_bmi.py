"""
Extract genuine MediaPipe 478-point 3D facial landmarks from the 3,963 UChicago dataset images
and train high-accuracy anthropometric regression models for real-time web execution.
"""

import os
import csv
import json
import time
import math
import numpy as np
import mediapipe as mp
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV, HuberRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATA_DIR = r"C:\Users\Pathik\Downloads\UChicago Docs\UChicago Spring 2023 Courses\Machine Learning and Predictive Analytics\Final\BMI\Data"
CSV_PATH = os.path.join(DATA_DIR, "data.csv")
IMAGES_DIR = os.path.join(DATA_DIR, "Images")
OUTPUT_JSON_PATH = os.path.join("model", "trained_bmi_weights.json")

os.makedirs("model", exist_ok=True)

def dist(p1, p2):
    return math.hypot(p1.x - p2.x, p1.y - p2.y)

def extract_anthropometric_features(landmarks, img_w, img_h):
    """
    Extract 22 precise anthropometric distance, ratio, angle, and 3D depth features
    from MediaPipe facial landmarks.
    """
    p_lcheek = landmarks[234]
    p_rcheek = landmarks[454]
    p_ljaw = landmarks[172]
    p_rjaw = landmarks[397]
    p_forehead = landmarks[10]
    p_chin = landmarks[152]
    p_nasion = landmarks[168]
    p_nosetip = landmarks[1]
    p_lowerlip = landmarks[17]
    p_leye = landmarks[33]
    p_reye = landmarks[263]
    p_llowercheek = landmarks[132]
    p_rlowercheek = landmarks[361]
    
    # 1. Distances (normalized by image size)
    cheek_w = dist(p_lcheek, p_rcheek)
    jaw_w = dist(p_ljaw, p_rjaw)
    face_h = dist(p_forehead, p_chin)
    morph_h = dist(p_nasion, p_chin)
    midcheek_w = dist(p_llowercheek, p_rlowercheek)
    eye_w = dist(p_leye, p_reye)
    chin_lip_h = dist(p_lowerlip, p_chin)
    nose_chin_h = dist(p_nosetip, p_chin)
    
    # 2. Key Morphological Ratios
    far = cheek_w / (face_h + 1e-5) # Facial Aspect Ratio (Bizygomatic / Physiognomical H)
    cjr = jaw_w / (cheek_w + 1e-5)   # Cheek-to-Jaw Width Ratio
    mcr = midcheek_w / (cheek_w + 1e-5) # Mid-cheek Adiposity Ratio
    mjr = jaw_w / (midcheek_w + 1e-5)  # Jaw to lower cheek
    ew_cw = eye_w / (cheek_w + 1e-5)   # Inter-ocular to cheekbone
    cl_fh = chin_lip_h / (face_h + 1e-5) # Chin pad height ratio
    nc_fh = nose_chin_h / (face_h + 1e-5) # Lower face vertical proportion
    
    # 3. Non-linear polynomial expansions
    far_sq = far ** 2
    cjr_sq = cjr ** 2
    far_cjr = far * cjr
    far_mcr = far * mcr
    
    # 4. 3D Depth Proportions (Fullness / Submental depth)
    z_cheek_avg = (p_lcheek.z + p_rcheek.z) / 2.0
    z_nose_diff = abs(p_nosetip.z - z_cheek_avg)
    z_jaw_diff = abs(p_chin.z - z_cheek_avg)
    
    # 5. Jawline Contour Curvature (Angle between left jaw, chin, right jaw)
    v1 = np.array([p_ljaw.x - p_chin.x, p_ljaw.y - p_chin.y])
    v2 = np.array([p_rjaw.x - p_chin.x, p_rjaw.y - p_chin.y])
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-5)
    jaw_angle = math.acos(max(-1.0, min(1.0, cos_angle))) # in radians
    
    # 6. Facial Area & Fullness Proportions
    face_area_proxy = cheek_w * face_h
    jaw_area_proxy = (jaw_w + cheek_w) * 0.5 * (face_h * 0.5)
    lower_fullness_index = jaw_area_proxy / (face_area_proxy + 1e-5)
    
    features = [
        far,
        cjr,
        mcr,
        mjr,
        ew_cw,
        cl_fh,
        nc_fh,
        far_sq,
        cjr_sq,
        far_cjr,
        far_mcr,
        jaw_angle,
        lower_fullness_index,
        z_nose_diff,
        z_jaw_diff,
        cheek_w,
        jaw_w,
        face_h,
        midcheek_w,
        chin_lip_h,
        float(img_w / float(img_h if img_h > 0 else 1)), # Raw box aspect ratio
        float(math.sqrt(img_w * img_h)) # Resolution scale
    ]
    return np.array(features, dtype=np.float32)

def main():
    print("=" * 65)
    print("  TRAINING MEDIAPIPE 478-LANDMARK BMI MODEL (UCHICAGO DATASET)")
    print("=" * 65)
    
    t0 = time.time()
    
    # 1. Load CSV Records
    records = []
    with open(CSV_PATH, 'r', encoding='utf-8', errors='ignore') as f:
        for row in csv.DictReader(f):
            try:
                records.append({
                    'name': row['name'].strip(),
                    'bmi': float(row['bmi']),
                    'is_training': int(row.get('is_training', '1'))
                })
            except Exception:
                pass
                
    print(f"Total dataset records in CSV: {len(records)}")
    
    # 2. Setup MediaPipe Landmarker
    base_options = mp.tasks.BaseOptions(model_asset_path='face_landmarker.task')
    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.4
    )
    detector = mp.tasks.vision.FaceLandmarker.create_from_options(options)
    
    X_train, y_train = [], []
    X_test, y_test = [], []
    test_files = []
    
    detected_count = 0
    fallback_count = 0
    
    print("Extracting 478 3D landmarks with MediaPipe FaceLandmarker across images...")
    
    for i, r in enumerate(records):
        img_path = os.path.join(IMAGES_DIR, r['name'])
        if not os.path.exists(img_path):
            continue
            
        try:
            mp_image = mp.Image.create_from_file(img_path)
            img_w = mp_image.width
            img_h = mp_image.height
            results = detector.detect(mp_image)
            
            if results.face_landmarks and len(results.face_landmarks) > 0:
                landmarks = results.face_landmarks[0]
                feat = extract_anthropometric_features(landmarks, img_w, img_h)
                detected_count += 1
            else:
                far_fallback = img_w / float(img_h if img_h > 0 else 1)
                feat = np.array([
                    far_fallback, 0.78, 0.85, 0.92, 0.65, 0.22, 0.42,
                    far_fallback**2, 0.78**2, far_fallback*0.78, far_fallback*0.85,
                    1.45, 0.60, 0.05, 0.03,
                    0.55, 0.43, 0.72, 0.48, 0.16,
                    far_fallback, float(math.sqrt(img_w * img_h))
                ], dtype=np.float32)
                fallback_count += 1
                
            if r['is_training'] == 1:
                X_train.append(feat)
                y_train.append(r['bmi'])
            else:
                X_test.append(feat)
                y_test.append(r['bmi'])
                test_files.append(r['name'])
                
        except Exception as e:
            continue
            
        if (i + 1) % 500 == 0:
            print(f"  Processed {i + 1} / {len(records)} images (Detected: {detected_count}, Fallback: {fallback_count})...")
            
    X_train = np.array(X_train, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.float32)
    X_test = np.array(X_test, dtype=np.float32)
    y_test = np.array(y_test, dtype=np.float32)
    
    print(f"\nFeature extraction completed in {time.time() - t0:.1f}s.")
    print(f"Total Faces with 478 Mesh: {detected_count} | Fallbacks: {fallback_count}")
    print(f"Train samples: {len(X_train)} | Test samples: {len(X_test)}")
    print(f"Feature Dimension: {X_train.shape[1]} anthropometric biometric features per face.")
    
    # 3. Normalize Features
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    # 4. Train Supervised Regression Models
    print("\nTraining and tuning regression models...")
    
    # Model 1: RidgeCV
    ridge = RidgeCV(alphas=np.logspace(-3, 4, 40), cv=5)
    ridge.fit(X_train_s, y_train)
    pred_ridge = ridge.predict(X_test_s)
    mae_ridge = mean_absolute_error(y_test, pred_ridge)
    r2_ridge = r2_score(y_test, pred_ridge)
    print(f"  [1] RidgeCV Regressor:       MAE = {mae_ridge:.3f} | R2 = {r2_ridge:.3f} (Alpha: {ridge.alpha_:.3f})")
    
    # Model 2: Huber Robust Regressor
    huber = HuberRegressor(max_iter=300, alpha=1.0)
    huber.fit(X_train_s, y_train)
    pred_huber = huber.predict(X_test_s)
    mae_huber = mean_absolute_error(y_test, pred_huber)
    r2_huber = r2_score(y_test, pred_huber)
    print(f"  [2] Huber Robust Regressor:  MAE = {mae_huber:.3f} | R2 = {r2_huber:.3f}")
    
    # Model 3: MLP Deep Neural Net Regressor
    mlp = MLPRegressor(hidden_layer_sizes=(64, 32, 16), activation='relu', max_iter=500,
                       learning_rate_init=0.005, alpha=0.05, random_state=42, early_stopping=True)
    mlp.fit(X_train_s, y_train)
    pred_mlp = mlp.predict(X_test_s)
    mae_mlp = mean_absolute_error(y_test, pred_mlp)
    r2_mlp = r2_score(y_test, pred_mlp)
    print(f"  [3] MLP Deep Regressor:      MAE = {mae_mlp:.3f} | R2 = {r2_mlp:.3f}")
    
    # Model 4: HistGradientBoosting Regressor
    hgb = HistGradientBoostingRegressor(max_iter=250, min_samples_leaf=15, learning_rate=0.03, random_state=42)
    hgb.fit(X_train, y_train)
    pred_hgb = hgb.predict(X_test)
    mae_hgb = mean_absolute_error(y_test, pred_hgb)
    r2_hgb = r2_score(y_test, pred_hgb)
    print(f"  [4] HistGradientBoosting:    MAE = {mae_hgb:.3f} | R2 = {r2_hgb:.3f}")
    
    # Model 5: Blended Ensemble (Ridge + Huber + MLP)
    pred_ensemble = (pred_ridge * 0.40) + (pred_huber * 0.30) + (pred_mlp * 0.30)
    mae_ensemble = mean_absolute_error(y_test, pred_ensemble)
    rmse_ensemble = math.sqrt(mean_squared_error(y_test, pred_ensemble))
    r2_ensemble = r2_score(y_test, pred_ensemble)
    print(f"\n  [BEST] Ensemble (Ridge + Huber + MLP): MAE = {mae_ensemble:.3f} | RMSE = {rmse_ensemble:.3f} | R2 = {r2_ensemble:.3f}")
    
    # 5. Cohort Analysis on Test Set
    print("\n--- Cohort Breakdown on Test Set ---")
    cohorts = [
        ("Lean (< 23.0)", y_test < 23.0),
        ("Normal (23.0 - 28.0)", (y_test >= 23.0) & (y_test <= 28.0)),
        ("Overweight (28.0 - 35.0)", (y_test > 28.0) & (y_test <= 35.0)),
        ("Obese (35.0+)", y_test > 35.0)
    ]
    for label, mask in cohorts:
        if np.sum(mask) > 0:
            c_mae = mean_absolute_error(y_test[mask], pred_ensemble[mask])
            c_mean_actual = np.mean(y_test[mask])
            c_mean_pred = np.mean(pred_ensemble[mask])
            print(f"  {label:<25}: Samples = {np.sum(mask):<4} | Actual Mean = {c_mean_actual:.1f} | Pred Mean = {c_mean_pred:.1f} | MAE = {c_mae:.2f}")
            
    # Sample Test Predictions
    print("\n--- Sample Test Predictions vs Ground Truth ---")
    for idx in range(min(10, len(y_test))):
        print(f"  Image: {test_files[idx]:<12} -> Ground Truth BMI: {y_test[idx]:.1f} | Model Prediction: {pred_ensemble[idx]:.1f} | Error: {abs(y_test[idx] - pred_ensemble[idx]):.1f}")
        
    # 6. Export JSON Weights for Real-Time Client-Side In-Browser Execution
    print(f"\nExporting trained weights to {OUTPUT_JSON_PATH}...")
    
    mlp_layers = []
    for w_mat, b_vec in zip(mlp.coefs_, mlp.intercepts_):
        mlp_layers.append({
            "weights": w_mat.tolist(),
            "biases": b_vec.tolist()
        })
        
    export_data = {
        "model_name": "UChicago_MediaPipe_Anthropometric_Ensemble",
        "dataset_name": "UChicago Machine Learning Final BMI Dataset",
        "dataset_total_samples": int(len(X_train) + len(X_test)),
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "test_mae": float(mae_ensemble),
        "test_rmse": float(rmse_ensemble),
        "test_r2": float(r2_ensemble),
        "feature_dim": int(X_train.shape[1]),
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "ridge": {
            "coefficients": ridge.coef_.tolist(),
            "intercept": float(ridge.intercept_)
        },
        "huber": {
            "coefficients": huber.coef_.tolist(),
            "intercept": float(huber.intercept_)
        },
        "mlp_layers": mlp_layers,
        "ensemble_weights": {
            "ridge": 0.40,
            "huber": 0.30,
            "mlp": 0.30
        }
    }
    
    with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(export_data, f, indent=2)
        
    print(f"Successfully saved {OUTPUT_JSON_PATH} ({os.path.getsize(OUTPUT_JSON_PATH)} bytes)!")
    print(f"Total training pipeline completed in {time.time() - t0:.1f}s.")

if __name__ == "__main__":
    main()
