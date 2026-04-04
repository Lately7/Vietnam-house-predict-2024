import pandas as pd
import numpy as np
import joblib

# import function từ file khác
from train_model import train_pipeline
from clean_data import clean_data_pipeline   # nếu bạn có file này


# =========================
# SAMPLE INPUT
# =========================
def sample_input():
    return {
        "Area": 80,
        "Frontage": 5,
        "Access_Road": 4,
        "Floors": 3,
        "Bedrooms": 3,
        "Bathrooms": 2
    }


# =========================
# BUILD INPUT (giống predict.py)
# =========================
def build_input_data(data):
    Area = data["Area"]
    Frontage = data["Frontage"]
    Access_Road = data["Access_Road"]
    Floors = data["Floors"]
    Bedrooms = data["Bedrooms"]
    Bathrooms = data["Bathrooms"]

    return {
        "Area": Area,
        "Frontage": Frontage,
        "Access_Road": Access_Road,
        "Floors": Floors,
        "Bedrooms": Bedrooms,
        "Bathrooms": Bathrooms,
        "Frontage_missing": 0,
        "Access_Road_missing": 0,
        "Floors_missing": 0,
        "Bedrooms_missing": 0,
        "Bathrooms_missing": 0,
        "Area_per_Bedroom": Area / (Bedrooms + 1),
        "Area_per_Bathroom": Area / (Bathrooms + 1),
        "Frontage_Area_ratio": Frontage / (Area + 1)
    }


# =========================
# PREPARE DF
# =========================
def prepare_df(data_dict, feature_cols):
    df = pd.DataFrame([data_dict])
    df = df.reindex(columns=feature_cols, fill_value=0)
    return df


# =========================
# MAIN PIPELINE
# =========================
if __name__ == "__main__":

    print("🚀 RUN FULL PIPELINE")

    # 1. CLEAN DATA
    print("\n[1] Cleaning data...")
    df_clean = clean_data_pipeline()   # nếu bạn có

    # 2. TRAIN MODEL
    print("\n[2] Training model...")
    model, feature_cols = train_pipeline()

    # 3. SAVE
    print("\n[3] Saving model...")
    joblib.dump(model, "best_house_price_model.pkl")
    joblib.dump(feature_cols, "model_features.pkl")

    # 4. PREDICT DEMO
    print("\n[4] Predict demo...")

    sample = sample_input()
    data = build_input_data(sample)
    df = prepare_df(data, feature_cols)

    pred_log = model.predict(df)[0]
    pred_price = np.expm1(pred_log)

    print("\nInput:", sample)
    print(f"💰 Giá dự đoán: {pred_price:.2f} tỷ VND")
    print(f"≈ {pred_price * 1_000_000_000:,.0f} VND")

    print("\n✅ DONE FULL PIPELINE")