def generate_price_explanation_prompt(
    brand,
    model,
    year,
    transmission,
    fuel_type,
    body_type,
    seating_capacity,
    region,
    mileage,
    variant,
    estimated_price
):
    prompt = f"""
Mobil bekas dengan spesifikasi berikut:

• Merek            : {brand}
• Model            : {model}
• Varian           : {variant}
• Tahun produksi   : {year}
• Transmisi        : {transmission}
• Jenis bahan bakar: {fuel_type}
• Tipe bodi        : {body_type}
• Kapasitas kursi  : {seating_capacity} penumpang
• Lokasi           : {region}
• Jarak tempuh     : {mileage:,} km

Diperkirakan memiliki harga sebesar **Rp {estimated_price:,.0f}**.

Jelaskan secara ringkas dan profesional (maksimal 250 kata) mengapa harga tersebut masuk akal berdasarkan informasi di atas. Sertakan pertimbangan umum seperti usia kendaraan, jenis transmisi, varian kendaraan, kondisi pasar mobil bekas, serta pengaruh wilayah terhadap harga jual.
"""
    return prompt.strip()