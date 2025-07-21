import pandas as pd
import numpy as np
import time
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

# === Konfigurasi ===
CURRENT_YEAR = 2025
MIN_MODEL_OCCURRENCE = 5
FOLDER = "legoas_v2/model"

os.makedirs(FOLDER, exist_ok=True)

DATA_FILE = os.path.join(FOLDER, 'train_dataset.csv')
MODEL_FILE = os.path.join(FOLDER, 'xgb_price_predictor.joblib')
COLUMNS_FILE = os.path.join(FOLDER, 'xgb_model_columns.joblib')
TEST_SET_FILE = os.path.join(FOLDER, 'xgb_test_set.joblib')
TARGET_COLUMN = 'price'

def preprocess_data(df):
    print("📦 Memulai preprocessing...")

    # 1. Rename kolom agar konsisten
    df = df.rename(columns={
        'tahun': 'vehicleModelDate',
        'km': 'value_mileage',
        'transmisi': 'transmision',
        'lokasi': 'addressRegion',
        'tipe_body': 'bodyType',
        'bahan_bakar': 'fuelType',
        'harga': 'price'
    })

    # 2. Hapus kolom yang tidak digunakan
    columns_to_drop = ['judul', 'kapasitas', 'warna', 'penjual', 'jenis']
    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns], errors='ignore')

    # 3. Bersihkan harga
    df['price'] = df['price'].astype(str).str.replace(r'[^\d]', '', regex=True).astype(float)

    # 4. Filter harga ekstrem
    q1, q3 = df['price'].quantile([0.05, 0.95])
    df = df[(df['price'] >= q1) & (df['price'] <= q3)]

    # 5. Konversi data numerik
    df['value_mileage'] = pd.to_numeric(df['value_mileage'], errors='coerce')
    df['vehicleModelDate'] = pd.to_numeric(df['vehicleModelDate'], errors='coerce')

    # 6. Tambah fitur turunan
    df['age'] = CURRENT_YEAR - df['vehicleModelDate']

    # 7. Filter model langka
    model_counts = df['model'].value_counts()
    common_models = model_counts[model_counts >= MIN_MODEL_OCCURRENCE].index
    df = df[df['model'].isin(common_models)]
    print(f"🧹 Model langka dibuang: {len(model_counts) - len(common_models)}")

    # 8. Feature engineering
    df['mileage_per_year'] = df['value_mileage'] / (df['age'] + 1)
    df['age_squared'] = df['age'] ** 2
    df['mileage_squared'] = df['value_mileage'] ** 2

    # 9. Transform harga (log)
    df['log_price'] = np.log1p(df['price'])

    # 10. Tambahkan kolom kategorikal
    categorical_cols = ['merek', 'model', 'varian', 'bodyType', 'fuelType', 'addressRegion', 'transmision']

    print(f"✅ Data siap. Total: {len(df)} baris\n")
    return df, categorical_cols

def train_xgb():
    print("🚗 Memulai training XGBoost...\n")

    df = pd.read_csv(DATA_FILE, low_memory=False)
    df, categorical_cols = preprocess_data(df)

    X = df.drop(columns=[TARGET_COLUMN, 'log_price'])
    y = df['log_price']

    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer([
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_cols)
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=42
    )

    print("🔧 Preprocessing data...")
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    regressor = XGBRegressor(
        n_estimators=1500,
        max_depth=8,
        learning_rate=0.03,
        subsample=0.7,
        colsample_bytree=0.7,
        random_state=42,
        n_jobs=-1,
        gamma=0.1,
        reg_alpha=0.01,
        reg_lambda=1,
        early_stopping_rounds=50,
        eval_metric='mae'
    )

    start = time.time()
    regressor.fit(X_train_processed, y_train,
                  eval_set=[(X_test_processed, y_test)],
                  verbose=False)
    end = time.time()

    y_pred_log = regressor.predict(X_test_processed)
    y_pred = np.expm1(y_pred_log)
    y_true = np.expm1(y_test)

    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    print(f"✅ Training selesai dalam {end - start:.2f} detik.")
    print(f"📉 MAE  : Rp {mae:,.2f}")
    print(f"📈 R²   : {r2 * 100:.2f}%")
    print(f"📊 MAPE : {mape:.2f}%\n")

    print("💾 Menyimpan model dan metadata...")
    full_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', regressor)
    ])
    joblib.dump(full_pipeline, MODEL_FILE)

    joblib.dump({
        'features': list(X.columns),
        'categorical': categorical_cols
    }, COLUMNS_FILE)

    joblib.dump((X_test, y_true), TEST_SET_FILE)
    print("✅ Semua file berhasil disimpan\n")

if __name__ == "__main__":
    train_xgb()
