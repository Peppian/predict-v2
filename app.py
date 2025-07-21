import streamlit as st
import pandas as pd
import joblib
import numpy as np

from ai_response import ask_openrouter
from prompt import generate_price_explanation_prompt

# === Load model dan metadata ===
MODEL_PATH = "legoas_v2/model/xgb_price_predictor.joblib"
COLUMNS_PATH = "legoas_v2/model/xgb_model_columns.joblib"
DATA_PATH = "legoas_v2/model/train_dataset.csv"
REFERENCE_PATH = "legoas_v2/model/validate_dataset.csv"

model = joblib.load(MODEL_PATH)
columns_meta = joblib.load(COLUMNS_PATH)
data = pd.read_csv(DATA_PATH)
ref_prices = pd.read_csv(REFERENCE_PATH)

# Bersihkan kolom harga agar bisa dihitung
for col in ['avg_price', 'low_price', 'mid_price', 'new_price']:
    ref_prices[col] = (
        ref_prices[col]
        .astype(str)
        .str.replace(",", "", regex=False)
        .str.strip()
        .astype(float)
    )

categorical_cols = columns_meta["categorical"]

st.set_page_config(page_title="Estimasi Harga Mobil Bekas", page_icon="🚘")
st.title("🚘 Estimasi Harga Mobil Bekas")

# === INPUT UI ===
brand_list = sorted(data['merek'].dropna().unique())
brand = st.selectbox("Pilih Merek Mobil", ["-"] + brand_list)

filtered_models = data[data['merek'] == brand]['model'].unique() if brand != "-" else []
model_selected = st.selectbox("Pilih Tipe / Model", ["-"] + list(filtered_models))

filtered_variants = data[
    (data['merek'] == brand) & (data['model'] == model_selected)
]['varian'].dropna().unique() if model_selected != "-" else []
variant_selected = st.selectbox("Pilih Varian", ["-"] + list(filtered_variants))

# Auto metadata
row = data[
    (data['merek'] == brand) &
    (data['model'] == model_selected) &
    (data['varian'] == variant_selected)
].head(1)

if not row.empty:
    body_type = row['tipe_body'].values[0]
    fuel_type = row['bahan_bakar'].values[0]
    seating_capacity = row['kapasitas'].values[0] if 'kapasitas' in row.columns else 0
else:
    body_type = "-"
    fuel_type = "-"
    seating_capacity = 0

# Auto display
st.markdown(f"**Body Type (auto):** {body_type}")
st.markdown(f"**Bahan Bakar (auto):** {fuel_type}")

region = st.selectbox("Wilayah", sorted(data['lokasi'].dropna().unique()))
transmission = st.selectbox("Transmisi", sorted(data['transmisi'].dropna().unique()))
year = st.slider("Tahun Produksi", int(data['tahun'].min()), 2025, 2020)
mileage = st.number_input("Kilometer", min_value=0, value=0)

# === PREDIKSI HARGA ===
if st.button("🔍 Estimasi Harga"):
    if "-" in [brand, model_selected, variant_selected, region, transmission, fuel_type, body_type]:
        st.warning("Mohon lengkapi semua input terlebih dahulu.")
    else:
        age = 2025 - year
        mileage_per_year = mileage / (age + 1)

        input_data = pd.DataFrame([{
            "value_mileage": mileage,
            "vehicleModelDate": year,
            "age": age,
            "mileage_per_year": mileage_per_year,
            "age_squared": age ** 2,
            "mileage_squared": mileage ** 2,
            "merek": brand,
            "model": model_selected,
            "varian": variant_selected,
            "bodyType": body_type,
            "fuelType": fuel_type,
            "addressRegion": region,
            "transmision": transmission
        }])

        # Pastikan semua kolom kategori ada
        for col in categorical_cols:
            if col not in input_data.columns:
                input_data[col] = ""

        log_price_pred = model.predict(input_data)[0]
        estimated_price = np.expm1(log_price_pred)

        # Ambil referensi harga
        ref_row = ref_prices[
            (ref_prices['brand'] == brand) &
            (ref_prices['model'] == model_selected) &
            (ref_prices['type'] == variant_selected) &
            (ref_prices['year'] == year) &
            (ref_prices['transmisi'] == transmission)
        ]

        if not ref_row.empty:
            low_price = ref_row['low_price'].values[0]
            mid_price = ref_row['mid_price'].values[0]

            # Koreksi harga
            if estimated_price > mid_price:
                estimated_price = mid_price
            elif estimated_price < low_price:
                estimated_price = max(estimated_price, low_price * 0.95)

        # Pembulatan ke kelipatan 1 juta
        estimated_price = round(estimated_price / 100_000) * 100_000

        # Tampilkan estimasi
        st.success(f"💰 Estimasi Harga: **Rp {estimated_price:,.0f}**")

        # Prompt untuk AI
        prompt_text = generate_price_explanation_prompt(
            brand=brand,
            model=model_selected,
            year=year,
            transmission=transmission,
            fuel_type=fuel_type,
            body_type=body_type,
            seating_capacity=seating_capacity,
            region=region,
            mileage=mileage,
            variant=variant_selected,
            estimated_price=estimated_price
        )

        with st.spinner("🔎 Menganalisis estimasi harga dengan AI..."):
            try:
                explanation = ask_openrouter(prompt_text)
                st.markdown("---")
                st.markdown("### 🧠 Penjelasan AI")
                st.markdown(explanation)
            except Exception as e:
                st.error("❌ Gagal mendapatkan respons dari AI. Silakan cek koneksi atau API key.")
                st.exception(e)
