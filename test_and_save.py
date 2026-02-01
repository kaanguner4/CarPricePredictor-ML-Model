import pandas as pd
import numpy as np
import re
import warnings
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from catboost import CatBoostRegressor

warnings.filterwarnings("ignore")

# --- VERİ TEMİZLEME FONKSİYONLARI ---

def parse_money(x):
    if pd.isna(x): return np.nan
    s = re.sub(r"[^0-9.]", "", str(x))
    return float(s) if s else np.nan

def parse_engine(text):
    """Engine sütunundan HP, Litre ve Silindir bilgilerini ayıklar."""
    if pd.isna(text): return pd.Series([np.nan, np.nan, np.nan, 0, 0])
    t = str(text).upper()
    hp = float(re.search(r"(\d+(?:\.\d+)?)\s*HP", t).group(1)) if re.search(r"(\d+(?:\.\d+)?)\s*HP", t) else np.nan
    liters = float(re.search(r"(\d+(?:\.\d+)?)\s*L\b", t).group(1)) if re.search(r"(\d+(?:\.\d+)?)\s*L\b", t) else np.nan
    
    cyl = np.nan
    m = re.search(r"\b([3-9]|10|12)\s*CYL", t)
    if m: cyl = float(m.group(1))
    else:
        m = re.search(r"\bV(\d{1,2})\b", t)
        if m: cyl = float(m.group(1))
    
    turbo = 1 if "TURBO" in t else 0
    hybrid = 1 if "HYBRID" in t else 0
    return pd.Series([hp, liters, cyl, turbo, hybrid])

def trans_type(x):
    t = str(x).lower()
    if "manual" in t or "m/t" in t: return "manual"
    if "cvt" in t: return "cvt"
    return "automatic"

def train_and_save_model():
    print("🚀 Süreç başlıyor: Veri yükleniyor...")
    
    # Veri setini oku (Dosya yolunun doğru olduğundan emin ol)
    try:
        df = pd.read_csv("data/used_cars.csv")
    except FileNotFoundError:
        print("❌ Hata: 'data/used_cars.csv' bulunamadı!")
        return

    # --- FEATURE ENGINEERING ---
    df["price_num"] = df["price"].apply(parse_money)
    df["milage_num"] = df["milage"].apply(parse_money)
    df[["hp","liters","cylinders","is_turbo","is_hybrid"]] = df["engine"].apply(parse_engine)
    df["transmission_type"] = df["transmission"].apply(trans_type)
    df["age"] = 2026 - df["model_year"] # Güncel yıl 2026 olarak güncellendi

    # Aykırı değerleri (Outliers) temizle
    df = df[(df["price_num"] >= 2000) & (df["price_num"] <= 150000)].reset_index(drop=True)

    # Eksik değerleri doldur
    for col in ["milage_num","hp","liters","cylinders","age"]:
        df[col] = df[col].fillna(df[col].median())

    cat_cols = ["brand","model","fuel_type","ext_col","int_col","clean_title","accident","transmission_type"]
    for c in cat_cols:
        df[c] = df[c].fillna("Unknown").astype(str)

    # Özellik seçimi
    features = cat_cols + ["age","milage_num","hp","liters","cylinders","is_turbo","is_hybrid"]
    X = df[features]
    y = df["price_num"]

    # Target Transformation: Fiyat tahmini için Log dönüşümü şart
    y_log = np.log1p(y)

    # Veriyi böl
    X_train, X_test, y_train_log, y_test_log = train_test_split(X, y_log, test_size=0.2, random_state=42)

    # --- CATBOOST MODEL EĞİTİMİ ---
    print("🧠 CatBoost eğitiliyor (MAE loss kullanılıyor)...")
    cat_features_idx = [X.columns.get_loc(c) for c in cat_cols]
    
    model = CatBoostRegressor(
        iterations=2000, 
        learning_rate=0.05, 
        depth=8, 
        loss_function='MAE', 
        eval_metric='MAE',
        verbose=200 # Her 200 adımda bir ilerlemeyi göster
    )

    model.fit(
        X_train, y_train_log, 
        cat_features=cat_features_idx,
        eval_set=(X_test, y_test_log),
        use_best_model=True
    )

    # --- PERFORMANS ÖLÇÜMÜ ---
    preds_log = model.predict(X_test)
    preds_actual = np.expm1(preds_log)
    y_test_actual = np.expm1(y_test_log)

    mae = mean_absolute_error(y_test_actual, preds_actual)
    print(f"\n✅ EĞİTİM TAMAMLANDI!")
    print(f"📊 Final Test MAE: ${mae:,.2f}")

    # --- MODELİ KAYDET ---
    model.save_model("car_price_model.cbm")
    print("💾 Model 'car_price_model.cbm' olarak kaydedildi.")

if __name__ == "__main__":
    train_and_save_model()