import streamlit as st
import numpy as np
import joblib

# Lade das gespeicherte Modell
model = joblib.load("best_model.pkl")

# Falls das Modell die polynomiale Regression ist, lade auch den Transformer
try:
    poly_transform = joblib.load("poly_transform.pkl")
    is_polynomial = True
except FileNotFoundError:
    poly_transform = None
    is_polynomial = False

# Titel der App
st.title("Vorhersage der Druckfestigkeit von Beton")
st.write("""
Geben Sie die Werte für die verschiedenen Merkmale des Betons ein, 
und das Modell sagt die Druckfestigkeit voraus.
""")

# Erstellen von Slidern für die Features
cement = st.slider("Zement (kg/m³)", min_value=100, max_value=500, step=10, value=300)
slag = st.slider("Hüttensand (kg/m³)", min_value=0, max_value=200, step=10, value=100)
fly_ash = st.slider("Flugasche (kg/m³)", min_value=0, max_value=200, step=10, value=50)
water = st.slider("Wasser (kg/m³)", min_value=120, max_value=250, step=5, value=180)
superplasticizer = st.slider("Superplastifizierer (kg/m³)", min_value=0, max_value=30, step=1, value=10)
coarse_aggregate = st.slider("Grobe Gesteinskörnung (kg/m³)", min_value=800, max_value=1200, step=10, value=1000)
fine_aggregate = st.slider("Feine Gesteinskörnung (kg/m³)", min_value=500, max_value=1000, step=10, value=800)
age = st.slider("Alter (Tage)", min_value=1, max_value=365, step=1, value=28)

# Eingabewerte in ein Array umwandeln
features = np.array([[cement, slag, fly_ash, water, superplasticizer, coarse_aggregate, fine_aggregate, age]])

# Falls polynomiale Regression, Eingaben transformieren
if is_polynomial and poly_transform is not None:
    features = poly_transform.transform(features)

# Vorhersage berechnen
predicted_strength = model.predict(features)[0]

# Anzeige der Vorhersage
st.subheader("Vorhergesagte Druckfestigkeit:")
st.write(f"{predicted_strength:.2f} MPa")