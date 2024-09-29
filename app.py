import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

# Judul aplikasi
st.title('Data Ikan')

# Upload dataset
uploaded_file = st.file_uploader("Upload CSV", type="csv")
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    # Data Understanding
    st.subheader("Data Understanding")
    st.write("Menampilkan 5 baris pertama dari dataset:")
    st.write(data.head())

    st.write("Menampilkan informasi dataset:")
    st.write(data.info())

    st.write("Menampilkan jumlah baris dan kolom:")
    st.write(data.shape)

    st.write("Menampilkan seluruh data dalam bentuk tabel:")
    st.write(data)

    # Visualisasi: Boxplot
    st.subheader("Visualisasi: Boxplot")
    plt.figure(figsize=(10, 5))
    sns.boxplot(data=data[['length', 'weight', 'w_l_ratio']])
    plt.title("Boxplot untuk Data Ikan")
    st.pyplot(plt)

    # Analisis Outlier
    st.subheader("Analisis Outlier")
    Q1 = data[['length', 'weight', 'w_l_ratio']].quantile(0.25)
    Q3 = data[['length', 'weight', 'w_l_ratio']].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = data[((data[['length', 'weight', 'w_l_ratio']] < lower_bound) | 
                      (data[['length', 'weight', 'w_l_ratio']] > upper_bound)).any(axis=1)]

    st.write("Outliers dalam dataset:")
    st.write(outliers)

    st.subheader("Visualisasi Outlier")
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=data[['length', 'weight', 'w_l_ratio']])
    plt.title("Boxplot untuk Mengidentifikasi Outliers")
    st.pyplot(plt)

    # Data Preparation
    st.subheader("Data Preparation")
    
    # Menggunakan Label Encoding untuk kolom target
    label_encoder = LabelEncoder()
    data['species'] = label_encoder.fit_transform(data['species'])

    X = data.drop('species', axis=1)  # Menggunakan 'species' sebagai target
    y = data['species']  # Target yang sudah diencoding

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Prediksi
    y_pred = model.predict(X_test)

    # Evaluasi
    st.subheader("Hasil Prediksi dan Aktual")
    st.write("Prediksi:", y_pred)
    st.write("Aktual:", y_test.values)

    # Visualisasi Hasil Prediksi vs Aktual
    plt.figure(figsize=(10, 5))
    plt.scatter(y_test, y_pred)
    plt.xlabel('Aktual')
    plt.ylabel('Prediksi')
    plt.title('Hasil Prediksi vs Aktual')
    plt.axline((0, 0), slope=1, color='red', linestyle='--')  # Garis 45 derajat
    st.pyplot(plt)

    # Koefisien Model
    st.subheader("Koefisien Model")
    st.write(model.coef_)

    # R-squared
    r2 = r2_score(y_test, y_pred)
    st.write("R-squared:", r2)

    # MSE
    mse = mean_squared_error(y_test, y_pred)
    st.write("Mean Squared Error:", mse)

    # Visualisasi Heatmap
    st.subheader("Visualisasi: Heatmap Korelasi")
    plt.figure(figsize=(10, 8))
    sns.heatmap(data.corr(), annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Heatmap Korelasi')
    st.pyplot(plt)

    # Menampilkan Outlier di Data Preparation
    st.subheader("Outlier dalam Data Preparation")
    st.write("Menampilkan outliers berdasarkan metode IQR:")
    st.write(outliers)