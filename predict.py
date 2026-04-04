import joblib
import pandas as pd
import numpy as np

# =========================
# LOAD MODEL
# =========================
model = joblib.load("best_house_price_model.pkl")
feature_cols = joblib.load("model_features.pkl")


# =========================
# HELPER
# =========================
def clip_with_warning(name, value, low, high):
    if value < low:
        print(f"[⚠️] {name}={value} < {low} → bị nâng lên {low}")
        return low
    elif value > high:
        print(f"[⚠️] {name}={value} > {high} → bị giảm xuống {high}")
        return high
    return value


# =========================
# BUILD INPUT DATA
# =========================
def build_input_data(Area, Frontage, Access_Road, Floors, Bedrooms, Bathrooms):

    print("\n--- Kiểm tra input ---")

    # clip + cảnh báo
    Area = clip_with_warning("Area", Area, 3, 140)
    Frontage = clip_with_warning("Frontage", Frontage, 2.5, 6.5)
    Access_Road = clip_with_warning("Access_Road", Access_Road, 3.5, 7.5)
    Floors = clip_with_warning("Floors", Floors, 1, 7)
    Bedrooms = clip_with_warning("Bedrooms", Bedrooms, 1.5, 5.5)
    Bathrooms = clip_with_warning("Bathrooms", Bathrooms, 1.5, 5.5)

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

    # feature engineering
    data["Area_per_Bedroom"] = Area / (Bedrooms + 1)
    data["Area_per_Bathroom"] = Area / (Bathrooms + 1)
    data["Frontage_Area_ratio"] = Frontage / (Area + 1)

    return data


# =========================
# PREPARE DF
# =========================
def prepare_df(data_dict):
    df = pd.DataFrame([data_dict])
    df = df.reindex(columns=feature_cols, fill_value=0)
    return df


# =========================
# PREDICT
# =========================
def predict(data_dict):
    df = prepare_df(data_dict)
    pred_log = model.predict(df)[0]
    pred_price = np.expm1(pred_log)
    return pred_price


# =========================
# INPUT FROM KEYBOARD
# =========================
def input_from_keyboard():
    print("\nNhập thông tin nhà:")

    Area = float(input("Diện tích (m2): "))
    Frontage = float(input("Mặt tiền: "))
    Access_Road = float(input("Đường vào: "))
    Floors = float(input("Số tầng: "))
    Bedrooms = float(input("Số phòng ngủ: "))
    Bathrooms = float(input("Số phòng tắm: "))

    return build_input_data(
        Area, Frontage, Access_Road, Floors, Bedrooms, Bathrooms
    )


# =========================
# SAMPLE DATA
# =========================
def sample_data():
    return [
        {"Area": 80, "Frontage": 5, "Access_Road": 4, "Floors": 3, "Bedrooms": 3, "Bathrooms": 2},
        {"Area": 50, "Frontage": 4, "Access_Road": 3, "Floors": 2, "Bedrooms": 2, "Bathrooms": 1},
        {"Area": 120, "Frontage": 6, "Access_Road": 6, "Floors": 4, "Bedrooms": 4, "Bathrooms": 3},
    ]


# =========================
# MAIN
# =========================
if __name__ == "__main__":

    print("===== HOUSE PRICE PREDICT =====")
    print("1. Nhập tay")
    print("2. Chạy dữ liệu mẫu")

    choice = input("Chọn (1 hoặc 2): ")

    if choice == "1":
        data = input_from_keyboard()
        price = predict(data)

        print("\nInput sau xử lý:", data)
        print(f"Giá dự đoán: {price:.2f} tỷ VND")
        print(f"≈ {price * 1_000_000_000:,.0f} VND")

    elif choice == "2":
        samples = sample_data()

        for i, s in enumerate(samples):
            print(f"\n--- House {i+1} ---")
            print("Input gốc:", s)

            data = build_input_data(**s)
            price = predict(data)

            print("Input sau xử lý:", data)
            print(f"Giá dự đoán: {price:.2f} tỷ VND")
            print(f"≈ {price * 1_000_000_000:,.0f} VND")

    else:
        print("Sai lựa chọn!")