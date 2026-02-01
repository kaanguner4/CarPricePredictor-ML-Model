# 🚗 CarPricePredictor – ML-Based Used Car Price Estimation System (EN)

## Project Overview

CarPricePredictor is a complete machine learning deployment project that predicts the market value of used cars based on real-world vehicle attributes.  
The system combines a trained CatBoost regression model with an interactive Streamlit web application to provide instant car price estimations.

This project represents a full pipeline from raw dataset → preprocessing → model training → evaluation → saving the trained model → deploying a real-time prediction app.

The goal is to create a professional and user-friendly vehicle valuation tool similar to platforms like Kelley Blue Book or AutoTrader.

---

## ✅ What We Built in the Last Week

Over the last week, the project evolved step-by-step into a production-style application:

### 1. Dataset Preparation & Feature Engineering
- Loaded a real used-car dataset (used_cars.csv)
- Cleaned price and mileage columns by removing currency symbols and formatting
- Standardized model names
- Extracted meaningful features such as:
  - vehicle age
  - engine information
  - turbo presence
  - hybrid flag

---

### 2. Machine Learning Model Development

We selected CatBoostRegressor due to its strong performance with mixed categorical + numerical data.

Key aspects:
- Handles categorical variables effectively without heavy encoding
- Produces stable results on structured tabular car datasets

The target variable was transformed using logarithmic scaling to improve performance:

- Model predicts log(price)
- Final output is recovered using:

final_price = np.expm1(predicted_log)

---

### 3. Model Training, Evaluation & Exporting

A full training script (test_and_save.py) was used to:

- Split train/test data
- Train CatBoost regression
- Evaluate performance with MAE (Mean Absolute Error)
- Save the final trained model:

model.save_model("car_price_model.cbm")

This exported file is later used directly inside the web app.

---

## Streamlit Web Application

The prediction system is deployed through a professional UI built in Streamlit.

### Features of the Web Interface

Users enter vehicle information through structured sections:

### Basic Info
- Brand
- Model
- Model Year
- Mileage
- Fuel Type

### Appearance & Condition
- Exterior / Interior Color
- Clean Title status
- Accident history
- Transmission type

### Technical Details
- Horsepower
- Engine size (L)
- Cylinders
- Turbo presence

---

## Prediction Workflow (How It Works)

When the user clicks Calculate Price, the following pipeline runs:

### Step 1: Input Collection
The user inputs are collected through Streamlit forms.

### Step 2: Feature Dictionary Construction

user_input = {
    "brand": brand,
    "model": model_name,
    ...
    "age": 2026 - year,
    "is_turbo": int(is_turbo),
    "is_hybrid": ...
}

### Step 3: Conversion to DataFrame

CatBoost expects structured input:

input_df = pd.DataFrame([user_input])

### Step 4: Price Prediction

res_log = model.predict(input_df)[0]
final_price = np.expm1(res_log)

### Step 5: Display Results

The application outputs:

✅ Estimated price  
✅ Confidence range (±5%)

st.metric("Estimated Value", f"${final_price:,.2f}")

---

## Technologies Used

- CatBoostRegressor
- Streamlit
- Pandas
- NumPy
- GitHub Version Control

---

## Project Structure

CarPriceProject/
│ app.py                  → Streamlit prediction interface
│ car_price_model.cbm     → Trained CatBoost model
│ requirements.txt        → Dependency list
│ test_and_save.py        → Training + evaluation pipeline
│ data/used_cars.csv      → Dataset

---

## Running the Project Locally

### Install dependencies

pip install -r requirements.txt

### Launch the app

streamlit run app.py

Then open:

http://localhost:8501

---

## Deployment & GitHub Upload

The complete project was successfully version-controlled and pushed to GitHub through:

- Git initialization
- Commit structure
- Remote repository connection
- Authentication via GitHub Personal Access Token
- Final push to main branch

---

## Future Improvements

- Dynamic year calculation (remove fixed 2026)
- Dropdown-based categorical input normalization
- Better confidence intervals using MAE
- Deployment to Streamlit Cloud for public access
- Removing large binaries and using Git LFS if needed

---

## Author

Kaan Güner  
Computer Engineering Student – AI & Data Science Focus  
Project: CarPricePredictor ML Model





---
🚗 Car Price Predictor ML Model (TR)
Bu proje, ikinci el araçların piyasa değerini tahmin etmek için geliştirilmiş, makine öğrenmesi tabanlı bir web uygulamasıdır. CatBoost Regression algoritması kullanılarak eğitilen model, araç özelliklerine göre yüksek doğrulukta fiyat tahminleri sunar.

🌟 Özellikler
Gelişmiş ML Algoritması: Kategorik verilerle yüksek performans gösteren CatBoost kütüphanesi kullanılmıştır.

Kullanıcı Dostu Arayüz: Streamlit framework'ü ile oluşturulmuş interaktif ve modern bir web arayüzü.

Detaylı Özellik Analizi: Marka, model, yıl, yakıt türü, vites tipi ve kilometre gibi pek çok parametreye göre analiz.

Hızlı Tahmin: Eğitilmiş model (car_price_model.cbm) sayesinde anlık sonuç üretimi.

🛠️ Kullanılan Teknolojiler
Python (Ana programlama dili)

CatBoost (Gradient Boosting algoritması)

Streamlit (Web uygulama arayüzü)

Pandas & NumPy (Veri işleme ve analiz)

Scikit-learn (Veri ön işleme ve metrikler)

🚀 Kurulum ve Çalıştırma
Projeyi yerel bilgisayarınızda çalıştırmak için aşağıdaki adımları izleyebilirsiniz:

1. Depoyu Klonlayın

Bash
git clone https://github.com/kaanguner4/CarPricePredictor-ML-Model.git
cd CarPricePredictor-ML-Model
2. Gerekli Kütüphaneleri Yükleyin

Bash
pip install -r requirements.txt
3. Uygulamayı Başlatın

Bash
streamlit run app.py
📊 Model Hakkında
Model, data/ klasöründe bulunan veri seti üzerinde eğitilmiştir. test_and_save.py dosyası, modelin eğitim sürecini, performans metriklerini ve eğitilen modelin .cbm formatında kaydedilmesini sağlar.

📂 Dosya Yapısı
app.py: Streamlit web uygulamasının ana dosyası.

car_price_model.cbm: Eğitilmiş ve kullanıma hazır CatBoost modeli.

test_and_save.py: Model eğitimi ve test işlemlerini yürüten script.

requirements.txt: Projenin bağımlılık listesi.

data/: Veri setinin bulunduğu dizin.

Bu proje Kaan Güner tarafından geliştirilmiştir.
