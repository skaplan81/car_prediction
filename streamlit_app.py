import streamlit as st
import joblib
import pandas as pd

# Title of the application
st.title("Araba Tahmin Uygulaması")

# Header
st.header("Arabanızın Fiyatını Tahmin Edin")

# Subheader
st.subheader("Bu uygulamayı kullanarak arabınızın değerini öğrenin.")


st.image("jaguar.jpeg", caption="Araba", width=400)

st.write("İlgilendiğiniz aracın tahmini piyasa değerini aşağıda görebilirsiniz.")


columns = joblib.load("features_list.joblib")

# st.write(columns)

min_year = 2003   # get_min_year()
max_year = 2018   # get_max_year()
year = st.number_input("Yil:", min_value=min_year, max_value=max_year)

min_present_price = 0.1
max_present_price = 100.0
present_price = st.slider("Present_Price:", min_value=min_present_price, max_value=max_present_price)

min_km = 500
max_km = 500_000
km = st.slider("km:", min_value=min_km, max_value=max_km)


fuel_type = st.selectbox(
    'Yakıt tipi:',
    ['Benzin', 'Dizel', 'LPG'])

if fuel_type == "Benzin":
    fuel = 'Petrol'
elif fuel_type == 'Dizel':
    fuel = 'Diesel'
else:
    fuel = 'CNG'

st.write('You selected:', fuel_type)

seller_type = st.selectbox(
    'Owner:',
    ['Galeri', 'Sahibinden'])

if seller_type == 'Galeri':
    seller = 'Dealer'
elif seller_type ==  'Sahibinden':
    seller = 'Individual'


transmission = st.selectbox(
    'Vites:',
    ['Manual', 'Automatic'])


owner = st.selectbox(
    'Sahibi:',
    [0,1,3])

sample_one = [{
"Year":                 year,
"Present_Price":        present_price,
"Kms_Driven":           km,
"Fuel_Type":            fuel,
"Seller_Type":          seller,
"Transmission":         transmission,
"Owner":                owner
    }]


df_s = pd.DataFrame(sample_one)
st.dataframe(df_s)


df_s["Year"] = max_year - df_s["Year"]
df_s = pd.get_dummies(df_s).reindex(columns=columns, fill_value=0)

scaler = joblib.load(open("scaler.joblib","rb"))
model = joblib.load(open("xgb_model.joblib","rb"))
df_s = scaler.transform(df_s)

if st.button('Tahmin Yap!'):
    tahmin = round(model.predict(df_s)[0] * 10_000)

    st.write('Arabinizin tahimini degeri:', tahmin)
else:
    st.write('Goodbye')
