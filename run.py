import pandas as pd
import numpy as np
import joblib

from train_model import train_pipeline
from clean_data import clean_data_pipeline


# =========================
# INPUT MODE
# =========================
def get_user_input():
    print("\n=== Nhập thông tin nhà ===")

    Area = float(input("Diện tích (m2): "))
    Frontage = float(input("Mặt tiền (m): "))
    Access_Road = float(input("Đường vào (m): "))
    Floors = float(input("Số tầng: "))
    Bedrooms = float(input("Số phòng ngủ: "))
    Bathrooms = float(input("Số phòng tắm: "))

    return {
        "Area": Area,
        "Frontage": Frontage,
        "Access_Road": Access_Road,
        "Floors": Floors,
        "Bedrooms": Bedrooms,
        "Bathrooms": Bathrooms
    }


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
# BUILD INPUT
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


def prepare_df(data_dict, feature_cols):
    df = pd.DataFrame([data_dict])
    df = df.reindex(columns=feature_cols, fill_value=0)
    return df


# =========================
# MAIN
# =========================
if __name__ == "__main__":

    # =========================
    # CHỌN MODE
    # =========================
    print("\nChọn mode:")
    print("1 - Demo (dữ liệu mẫu)")
    print("2 - Nhập tay")

    choice = input("Nhập lựa chọn (1/2): ").strip()

    # =========================
    # 1. CLEAN DATA
    # =========================
    print("\n[1] Cleaning data...")
    df_raw, df_clean = clean_data_pipeline()

    # =========================
    # 2. TRAIN MODEL
    # =========================
    print("\n[2] Training model...")
    model, feature_cols = train_pipeline()

    # =========================
    # 3. SAVE
    # =========================
    print("\n[3] Saving model...")
    joblib.dump(model, "best_house_price_model.pkl")
    joblib.dump(feature_cols, "model_features.pkl")

    # =========================
    # 4. INPUT
    # =========================
    print("\n[4] Predict...")

    if choice == "2":
        user_input = get_user_input()
    else:
        print("\nDùng input mẫu")
        user_input = sample_input()

    # =========================
    # 5. PREDICT
    # =========================
    data = build_input_data(user_input)
    df = prepare_df(data, feature_cols)

    pred_log = model.predict(df)[0]
    pred_price = np.expm1(pred_log)

    print("\n==============================")
    print("Input:", user_input)
    print("Processed:", data)
    print(f"Giá dự đoán: {pred_price:.2f} tỷ VND")
    print(f"≈ {pred_price * 1_000_000_000:,.0f} VND")
    print("==============================")
