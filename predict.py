import joblib
import pandas as pd
import numpy as np


def load_artifacts(
    model_path="best_house_price_model.pkl",
    features_path="model_features.pkl"
):
    model = joblib.load(model_path)
    feature_cols = joblib.load(features_path)
    return model, feature_cols


def warn_range(name, value, low, high):
    if value < low or value > high:
        print(f"[WARNING] {name}={value} nằm ngoài khoảng train [{low}, {high}]")


def build_input_data(Area, Frontage, Access_Road, Floors, Bedrooms, Bathrooms):
    print("\n--- Kiểm tra input ---")

    warn_range("Area", Area, 3, 140)
    warn_range("Frontage", Frontage, 2.5, 6.5)
    warn_range("Access_Road", Access_Road, 3.5, 7.5)
    warn_range("Floors", Floors, 1, 7)
    warn_range("Bedrooms", Bedrooms, 1.5, 5.5)
    warn_range("Bathrooms", Bathrooms, 1.5, 5.5)

    data = {
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
    }

    data["Area_per_Bedroom"] = Area / (Bedrooms + 1)
    data["Area_per_Bathroom"] = Area / (Bathrooms + 1)
    data["Frontage_Area_ratio"] = Frontage / (Area + 1)

    return data


def prepare_df(data_dict, feature_cols):
    df = pd.DataFrame([data_dict])
    df = df.reindex(columns=feature_cols, fill_value=0)
    return df


def predict_price(data_dict, model, feature_cols):
    df = prepare_df(data_dict, feature_cols)
    pred_log = model.predict(df)[0]
    pred_price = np.expm1(pred_log)
    return pred_price


def run_prediction_demo():
    model, feature_cols = load_artifacts()

    sample = {
        "Area": 80,
        "Frontage": 5,
        "Access_Road": 4,
        "Floors": 3,
        "Bedrooms": 3,
        "Bathrooms": 2
    }

    data = build_input_data(**sample)
    price = predict_price(data, model, feature_cols)

    print("\nInput demo:", sample)
    print("Input sau xử lý:", data)
    print(f"Giá dự đoán: {price:.2f} tỷ VND")
    print(f"Tức khoảng: {price * 1_000_000_000:,.0f} VND")


if __name__ == "__main__":
    run_prediction_demo()