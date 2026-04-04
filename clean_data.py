import pandas as pd
import numpy as np

def clean_data_pipeline(
    input_path="vietnam_housing_dataset.csv",
    raw_output_path="vietnam_housing_cleaned_raw.csv",
    encoded_output_path="vietnam_housing_cleaned_encoded.csv"
):
    # Đọc file
    df = pd.read_csv(input_path)

    # Chuẩn hóa tên cột
    df.columns = [c.strip() for c in df.columns]

    rename_map = {
        "Address": "Address",
        "Area": "Area",
        "Frontage": "Frontage",
        "Access Road": "Access_Road",
        "House direction": "House_direction",
        "Balcony direction": "Balcony_direction",
        "Floors": "Floors",
        "Bedrooms": "Bedrooms",
        "Bathrooms": "Bathrooms",
        "Legal status": "Legal_status",
        "Furniture state": "Furniture_state",
        "Price": "Price"
    }
    df = df.rename(columns=rename_map)

    text_cols = ["Address", "House_direction", "Balcony_direction", "Legal_status", "Furniture_state"]

    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
            df[col] = df[col].replace({
                "nan": np.nan,
                "None": np.nan,
                "": np.nan,
                "  ": np.nan
            })

    num_cols = ["Area", "Frontage", "Access_Road", "Floors", "Bedrooms", "Bathrooms", "Price"]

    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df[(df["Price"].notna()) & (df["Price"] > 0)]
    df = df[(df["Area"].notna()) & (df["Area"] > 0)]

    df = df[df["Area"] <= 1000]
    df = df[df["Price"] <= df["Price"].quantile(0.995)]

    df = df.drop_duplicates()

    subset_dup = ["Address", "Area", "Price"]
    subset_dup = [c for c in subset_dup if c in df.columns]
    df = df.drop_duplicates(subset=subset_dup, keep="first")

    missing_ratio = df.isna().mean().sort_values(ascending=False)
    print("\nTỉ lệ dữ liệu bị thiếu:")
    print(missing_ratio)

    drop_cols = []
    for col in ["Balcony_direction", "House_direction"]:
        if col in df.columns and df[col].isna().mean() > 0.65:
            drop_cols.append(col)

    if "Address" in df.columns:
        drop_cols.append("Address")

    df = df.drop(columns=drop_cols, errors="ignore")

    print("\nCác cột đã bỏ:", drop_cols)

    candidate_missing_flag_cols = ["Frontage", "Access_Road", "Floors", "Bedrooms", "Bathrooms"]
    for col in candidate_missing_flag_cols:
        if col in df.columns:
            df[f"{col}_missing"] = df[col].isna().astype(int)

    numeric_cols_now = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_feature_cols = [c for c in numeric_cols_now if c != "Price"]

    for col in numeric_feature_cols:
        if df[col].isna().sum() > 0:
            df[col] = df[col].fillna(df[col].median())

    cat_cols_now = df.select_dtypes(include=["object"]).columns.tolist()
    for col in cat_cols_now:
        df[col] = df[col].fillna("Unknown")

    def clip_outliers_iqr(dataframe, cols, k=1.5):
        df_out = dataframe.copy()
        for col in cols:
            if col in df_out.columns:
                q1 = df_out[col].quantile(0.25)
                q3 = df_out[col].quantile(0.75)
                iqr = q3 - q1
                low = q1 - k * iqr
                high = q3 + k * iqr
                df_out[col] = df_out[col].clip(lower=low, upper=high)
        return df_out

    outlier_cols = ["Area", "Frontage", "Access_Road", "Floors", "Bedrooms", "Bathrooms"]
    outlier_cols = [c for c in outlier_cols if c in df.columns]
    df = clip_outliers_iqr(df, outlier_cols, k=1.5)

    if "Area" in df.columns and "Bedrooms" in df.columns:
        df["Area_per_Bedroom"] = df["Area"] / (df["Bedrooms"] + 1)

    if "Area" in df.columns and "Bathrooms" in df.columns:
        df["Area_per_Bathroom"] = df["Area"] / (df["Bathrooms"] + 1)

    if "Frontage" in df.columns and "Area" in df.columns:
        df["Frontage_Area_ratio"] = df["Frontage"] / (df["Area"] + 1)

    cat_cols_now = df.select_dtypes(include=["object"]).columns.tolist()
    df_clean = pd.get_dummies(df, columns=cat_cols_now, drop_first=True)

    df_clean["Price_log"] = np.log1p(df_clean["Price"])

    df.to_csv(raw_output_path, index=False)
    df_clean.to_csv(encoded_output_path, index=False)

    print("\nShape sau clean raw:", df.shape)
    print("Shape sau encode:", df_clean.shape)
    print("\nĐã lưu:")
    print("-", raw_output_path)
    print("-", encoded_output_path)

    return df, df_clean


if __name__ == "__main__":
    clean_data_pipeline()