import pandas as pd
import numpy as np
import os
import config

def load_data():
    df = pd.read_csv(config.RAW_DATA_FILE, low_memory=False)
    return df

def clean_data(df):
    df = df.drop_duplicates()
    df.columns = [c.strip().upper().replace(" ", "_") for c in df.columns]

    null_ratio = df.isnull().mean()
    drop_cols = null_ratio[null_ratio > 0.7].index
    df = df.drop(columns=drop_cols)

    critical_cols = ["LATITUDE", "LONGITUDE", "CRASH_DATE", "CRASH_TIME"]
    critical_cols = [c for c in critical_cols if c in df.columns]
    df = df.dropna(subset=critical_cols)

    # Kategorik sütunları fillna ile doldur
    cat_cols = df.select_dtypes(include="object").columns
    for c in cat_cols:
        df[c] = df[c].fillna("UNKNOWN")

    # İkinci araç sütunlarını "NONE" ile doldur
    second_vehicle_cols = ["CONTRIBUTING_FACTOR_VEHICLE_2", "VEHICLE_TYPE_CODE_2"]
    for c in second_vehicle_cols:
        if c in df.columns:
            df[c] = df[c].fillna("NONE")

    # Sayısal sütunları 0 ile doldur
    num_cols = [c for c in df.columns if df[c].dtype in [np.int64, np.float64]]
    for c in num_cols:
        df[c] = df[c].fillna(0)

    # NYC koordinat filtresi
    df = df[(df["LATITUDE"].between(40.4, 41.0)) & (df["LONGITUDE"].between(-74.3, -73.6))]

    # Önemli sütunları tut
    keep_cols = [
        "CRASH_DATE", "CRASH_TIME", "BOROUGH", "ZIP_CODE", "LATITUDE", "LONGITUDE",
        "NUMBER_OF_PERSONS_INJURED", "NUMBER_OF_PERSONS_KILLED",
        "NUMBER_OF_PEDESTRIANS_INJURED", "NUMBER_OF_PEDESTRIANS_KILLED",
        "NUMBER_OF_CYCLISTS_INJURED", "NUMBER_OF_CYCLISTS_KILLED",
        "NUMBER_OF_MOTORISTS_INJURED", "NUMBER_OF_MOTORISTS_KILLED",
        "CONTRIBUTING_FACTOR_VEHICLE_1", "CONTRIBUTING_FACTOR_VEHICLE_2",
        "VEHICLE_TYPE_CODE_1", "VEHICLE_TYPE_CODE_2"
    ]
    df = df[[c for c in keep_cols if c in df.columns]]

    # Tarih ve saat feature’ları
    df["CRASH_DATE"] = pd.to_datetime(df["CRASH_DATE"], errors="coerce")
    df = df.dropna(subset=["CRASH_DATE"])

    df["YEAR"] = df["CRASH_DATE"].dt.year
    df["MONTH"] = df["CRASH_DATE"].dt.month
    df["DAY_OF_WEEK"] = df["CRASH_DATE"].dt.dayofweek
    df["HOUR"] = pd.to_datetime(df["CRASH_TIME"], format="%H:%M", errors="coerce").dt.hour
    df = df.dropna(subset=["HOUR"])

    # Severity score
    df["SEVERITY_SCORE"] = (
        df["NUMBER_OF_PERSONS_KILLED"] * 3 +
        df["NUMBER_OF_PERSONS_INJURED"]
    )

    return df

def save_data(df):
    os.makedirs(config.DATA_DIR, exist_ok=True)
    df.to_csv(config.PROCESSED_DATA_FILE, index=False)

if __name__ == "__main__":
    df = load_data()
    df_clean = clean_data(df)
    save_data(df_clean)
