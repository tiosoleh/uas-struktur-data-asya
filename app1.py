import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

# =============================
# PAGE CONFIG
# =============================
st.set_page_config(
    page_title="Sistem Diagnosa COVID-19",
    page_icon="🧬",
    layout="centered"
)

# =============================
# STYLE
# =============================
st.markdown("""
<style>
body {
    background-color: #F5F7FA;
}
.card {
    background-color: white;
    padding: 25px;
    border-radius: 15px;
    box-shadow: 0 4px 10px rgba(0,0,0,0.1);
    margin-bottom: 20px;
}
.title {
    text-align: center;
    color: #0A3D62;
}
.subtitle {
    text-align: center;
    color: #576574;
}
</style>
""", unsafe_allow_html=True)

# =============================
# HEADER
# =============================
st.markdown("<h1 class='title'>🧬 Sistem Diagnosa COVID-19</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Decision Tree berbasis Umur & Gejala</p>", unsafe_allow_html=True)

st.markdown("---")

# =============================
# LOAD DATASET & TRAIN MODEL
# =============================
df = pd.read_excel("dataset_covid_umur_10_sampel.xlsx")

encoder = LabelEncoder()
for col in df.columns:
    df[col] = encoder.fit_transform(df[col])

X = df.drop("Diagnosa", axis=1)
y = df["Diagnosa"]

model = DecisionTreeClassifier(criterion="entropy")
model.fit(X, y)

# =============================
# INPUT FORM
# =============================
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.subheader("📝 Input Data Pasien")

umur = st.selectbox("Kategori Umur", ["Muda", "Dewasa", "Lansia"])
demam = st.radio("Demam", ["ya", "tidak"])
sesak = st.radio("Sesak Nafas", ["ya", "tidak"])

st.markdown("</div>", unsafe_allow_html=True)

# =============================
# MAPPING INPUT
# =============================
umur_map = {"Muda": 0, "Dewasa": 1, "Lansia": 2}
ya_tidak_map = {"tidak": 0, "ya": 1}

input_data = pd.DataFrame(
    [[umur_map[umur], ya_tidak_map[demam], ya_tidak_map[sesak]]],
    columns=X.columns
)

# =============================
# HASIL DIAGNOSA
# =============================
st.markdown("### 🔍 Hasil Diagnosa")

if st.button("🧪 Proses Diagnosa"):
    hasil = model.predict(input_data)

    if hasil[0] == 1:
        st.markdown("""
        <div class='card' style='border-left:8px solid #e84118'>
            <h2>❗ POSITIF COVID-19</h2>
            <p>Pasien memiliki indikasi COVID-19 berdasarkan data yang dimasukkan.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class='card' style='border-left:8px solid #44bd32'>
            <h2>✅ NEGATIF COVID-19</h2>
            <p>Tidak terdeteksi indikasi COVID-19 berdasarkan data yang dimasukkan.</p>
        </div>
        """, unsafe_allow_html=True)

# =============================
# FOOTER
# =============================
st.markdown("---")
st.markdown(
    "<p style='text-align:center;font-size:12px;color:gray;'>UAS Struktur Data | Decision Tree</p>",
    unsafe_allow_html=True
)
