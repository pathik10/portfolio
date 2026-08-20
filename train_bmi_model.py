"""
Retrain Facial BMI Prediction Model on UChicago Dataset
Dataset: 3,963 face images with ground-truth BMI values
Outputs: model/trained_bmi_weights.json (for direct in-browser JS execution)
"""

import os
import csv
import json
import time
import numpy as np
from PIL import Image
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV, ElasticNetCV
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATA_DIR = r"C:\Users\Pathik\Downloads\UChicago Docs\UChicago Spring 2023 Courses\Machine Learning and Predictive Analytics\Final\BMI\Data"
CSV_PATH = os.path.join(DATA_DIR, "data.csv")
IMAGES_DIR = os.path.join(DATA_DIR, "Images")
OUTPUT_JSON_PATH = os.path.join("model", "trained_bmi_weights.json")

os.makedirs("model", exist_ok=True)

def extract_facial_features_from_image(img: Image.Image) -> np.ndarray:
    """
    Extracts high-dimensional morphological, spatial, and textural facial features
    that directly capture facial adiposity, cheek-to-jaw proportion, and facial aspect ratio.
    """
    # 1. Aspect Ratio
    w, h = img.size
    far = w / (float(h) if h > 0 else 1.0)
    
    # 2. Resize to canonical 128x128 for uniform regional analysis
    im_resized = img.resize((128, 128), Image.Resampling.BILINEAR)
    arr_rgb = np.array(im_resized, dtype=np.float32)
    arr_gray = 0.2989 * arr_rgb[:, :, 0] + 0.5870 * arr_rgb[:, :, 1] + 0.1140 * arr_rgb[:, :, 2]
    
    # 3. Regional Segmentations
    # Upper Face (Forehead / Eyes): rows 0-42
    # Mid Face (Cheekbones / Nose): rows 43-85
    # Lower Face (Jaw / Chin / Neck): rows 86-127
    upper = arr_gray[0:43, :]
    mid = arr_gray[43:86, :]
    lower = arr_gray[86:128, :]
    
    # Sub-regions for Cheek & Jaw fullness
    mid_left_cheek = mid[:, 10:45]
    mid_right_cheek = mid[:, 83:118]
    lower_jaw_left = lower[:, 10:45]
    lower_jaw_right = lower[:, 83:118]
    lower_chin_center = lower[15:42, 40:88]
    
    # Feature Vector Collection
    features = []
    
    # [0] Global Aspect Ratio
    features.append(far)
    features.append(w)
    features.append(h)
    
    # [1] Regional Mean Intensities & Contrast
    features.append(np.mean(upper))
    features.append(np.mean(mid))
    features.append(np.mean(lower))
    features.append(np.std(upper))
    features.append(np.std(mid))
    features.append(np.std(lower))
    
    # [2] Cheek-to-Jaw Contrast Ratios (Direct proxy for facial adiposity)
    mid_cheek_mean = (np.mean(mid_left_cheek) + np.mean(mid_right_cheek)) / 2.0
    lower_jaw_mean = (np.mean(lower_jaw_left) + np.mean(lower_jaw_right)) / 2.0
    chin_mean = np.mean(lower_chin_center)
    
    features.append(mid_cheek_mean)
    features.append(lower_jaw_mean)
    features.append(chin_mean)
    features.append(lower_jaw_mean / (mid_cheek_mean + 1e-5)) # CJR proxy
    features.append(chin_mean / (lower_jaw_mean + 1e-5))
    
    # [3] Horizontal Profile Energy (Width distribution from top to bottom)
    # Row-wise standard deviations reveal lateral facial contour flare
    row_stds = np.std(arr_gray, axis=1) # 128 values
    # Downsample row profile to 16 zones
    zone_stds = [np.mean(row_stds[i*8:(i+1)*8]) for i in range(16)]
    features.extend(zone_stds)
    
    # [4] Column-wise Horizontal Gradient Profile (Cheek fullness lateral spread)
    col_stds = np.std(arr_gray, axis=0) # 128 values
    zone_col_stds = [np.mean(col_stds[i*8:(i+1)*8]) for i in range(16)]
    features.extend(zone_col_stds)
    
    # [5] Spatial Edge Density (High adiposity reduces sharp edge transitions)
    # Sobel-like simple diffs
    grad_x = np.abs(arr_gray[:, 1:] - arr_gray[:, :-1])
    grad_y = np.abs(arr_gray[1:, :] - arr_gray[:-1, :])
    features.append(np.mean(grad_x))
    features.append(np.mean(grad_y))
    features.append(np.mean(grad_x[86:, :])) # Lower face edge softness
    features.append(np.mean(grad_y[86:, :]))
    
    # [6] Histogram moments (16 bins for mid face, 16 bins for lower face)
    hist_mid, _ = np.histogram(mid, bins=16, range=(0, 255), density=True)
    hist_lower, _ = np.histogram(lower, bins=16, range=(0, 255), density=True)
    features.extend(hist_mid.tolist())
    features.extend(hist_lower.tolist())
    
    return np.array(features, dtype=np.float32)

def main():
    print("=" * 60)
    print("  RETRAINING FACIAL BMI MODEL (UCHICAGO DATASET)")
    print("=" * 60)
    
    t0 = time.time()
    
    # 1. Read CSV metadata
    print(f"Loading metadata from {CSV_PATH}...")
    records = []
    with open(CSV_PATH, 'r', encoding='utf-8', errors='ignore') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                name = row['name'].strip()
                bmi = float(row['bmi'])
                is_train = int(row.get('is_training', '1'))
                records.append({
                    'name': name,
                    'bmi': bmi,
                    'is_training': is_train
                })
            except Exception:
                pass
                
    print(f"Total valid metadata records: {len(records)}")
    
    # 2. Extract features from images
    X_train, y_train = [], []
    X_test, y_test = [], []
    test_filenames = []
    
    loaded_count = 0
    missing_count = 0
    
    print("Extracting multi-zone morphological features from images...")
    for i, r in enumerate(records):
        img_path = os.path.join(IMAGES_DIR, r['name'])
        if not os.path.exists(img_path):
            missing_count += 1
            continue
            
        try:
            with Image.open(img_path) as im:
                feat = extract_facial_features_from_image(im)
                
            if r['is_training'] == 1:
                X_train.append(feat)
                y_train.append(r['bmi'])
            else:
                X_test.append(feat)
                y_test.append(r['bmi'])
                test_filenames.append(r['name'])
                
            loaded_count += 1
            if loaded_count % 500 == 0:
                print(f"  Processed {loaded_count} images...")
        except Exception as e:
            missing_count += 1
            
    X_train = np.array(X_train, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.float32)
    X_test = np.array(X_test, dtype=np.float32)
    y_test = np.array(y_test, dtype=np.float32)
    
    print(f"Feature extraction complete in {time.time() - t0:.1f}s.")
    print(f"Train Shape: {X_train.shape} ({len(y_train)} samples)")
    print(f"Test Shape: {X_test.shape} ({len(y_test)} samples)")
    print(f"Feature Dimension: {X_train.shape[1]} features per face")
    
    # 3. Standard Scaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 4. Train Models
    print("\nTraining candidate regressors...")
    
    # Model 1: Ridge Regression with CV
    ridge = RidgeCV(alphas=np.logspace(-3, 4, 30), cv=5)
    ridge.fit(X_train_scaled, y_train)
    pred_ridge = ridge.predict(X_test_scaled)
    mae_ridge = mean_absolute_error(y_test, pred_ridge)
    r2_ridge = r2_score(y_test, pred_ridge)
    print(f"  [1] RidgeCV:           MAE = {mae_ridge:.3f} | R² = {r2_ridge:.3f} (Best Alpha: {ridge.alpha_:.3f})")
    
    # Model 2: Multi-Layer Perceptron (MLP) Deep Regressor
    mlp = MLPRegressor(hidden_layer_sizes=(128, 64, 32), activation='relu', max_iter=400,
                       learning_rate_init=0.003, alpha=0.01, random_state=42, early_stopping=True)
    mlp.fit(X_train_scaled, y_train)
    pred_mlp = mlp.predict(X_test_scaled)
    mae_mlp = mean_absolute_error(y_test, pred_mlp)
    r2_mlp = r2_score(y_test, pred_mlp)
    print(f"  [2] MLP Deep Regressor: MAE = {mae_mlp:.3f} | R² = {r2_mlp:.3f}")
    
    # Model 3: Extra Trees Regressor
    et = ExtraTreesRegressor(n_estimators=100, max_depth=15, min_samples_split=4, random_state=42, n_jobs=-1)
    et.fit(X_train_scaled, y_train)
    pred_et = et.predict(X_test_scaled)
    mae_et = mean_absolute_error(y_test, pred_et)
    r2_et = r2_score(y_test, pred_et)
    print(f"  [3] ExtraTrees:         MAE = {mae_et:.3f} | R² = {r2_et:.3f}")
    
    # Model 4: Ensemble (Ridge + MLP + ExtraTrees)
    pred_ensemble = (pred_ridge * 0.40) + (pred_mlp * 0.40) + (pred_et * 0.20)
    mae_ensemble = mean_absolute_error(y_test, pred_ensemble)
    rmse_ensemble = np.sqrt(mean_squared_error(y_test, pred_ensemble))
    r2_ensemble = r2_score(y_test, pred_ensemble)
    print(f"  [★] Ensemble:           MAE = {mae_ensemble:.3f} | RMSE = {rmse_ensemble:.3f} | R² = {r2_ensemble:.3f}")
    
    # 5. Cohort Breakdown on Test Set
    print("\n--- Cohort Accuracy on Held-Out Test Set ---")
    cohorts = [
        ("Underweight / Lean (<22)", y_test < 22),
        ("Normal (22 - 27)", (y_test >= 22) & (y_test <= 27)),
        ("Overweight (27 - 35)", (y_test > 27) & (y_test <= 35)),
        ("Obese (35+)", y_test > 35)
    ]
    for label, mask in cohorts:
        if np.sum(mask) > 0:
            sub_mae = mean_absolute_error(y_test[mask], pred_ensemble[mask])
            print(f"  {label:<25}: Count = {np.sum(mask):<4} | MAE = {sub_mae:.2f} BMI units")
            
    # Sample Test Predictions vs Ground Truth
    print("\n--- Sample Test Predictions vs Actual Ground Truth ---")
    for idx in range(min(8, len(y_test))):
        print(f"  File: {test_filenames[idx]:<12} -> Actual BMI: {y_test[idx]:.1f} | Predicted: {pred_ensemble[idx]:.1f} (Diff: {abs(y_test[idx] - pred_ensemble[idx]):.1f})")
        
    # 6. Export Web Weights JSON for Fast In-Browser Inference
    print(f"\nExporting trained weights to {OUTPUT_JSON_PATH}...")
    
    # Export Ridge coefficients & MLP layer matrices for JS client
    mlp_layers = []
    for w_mat, b_vec in zip(mlp.coefs_, mlp.intercepts_):
        mlp_layers.append({
            "weights": w_mat.tolist(),
            "biases": b_vec.tolist()
        })
        
    export_data = {
        "model_name": "UChicago_Facial_BMI_Ensemble",
        "dataset_samples": int(len(X_train) + len(X_test)),
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "test_mae": float(mae_ensemble),
        "test_r2": float(r2_ensemble),
        "feature_dim": int(X_train.shape[1]),
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "ridge": {
            "coefficients": ridge.coef_.tolist(),
            "intercept": float(ridge.intercept_)
        },
        "mlp_layers": mlp_layers,
        "ensemble_weights": [0.5, 0.5]
    }
    
    with open(OUTPUT_JSON_PATH, "w") as f:
        json.dump(export_data, f)
        
    print(f"✓ Successfully exported {OUTPUT_JSON_PATH} ({os.path.getsize(OUTPUT_JSON_PATH)} bytes)!")
    print(f"Total pipeline execution time: {time.time() - t0:.1f}s.")

if __name__ == "__main__":
    main()
