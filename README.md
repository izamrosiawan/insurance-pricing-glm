# Motor Insurance Pure Premium Modeling using GLM & Tweedie Regression

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-GLM-orange.svg)](https://scikit-learn.org/)
[![Domain](https://img.shields.io/badge/Domain-Actuarial%20Pricing-green.svg)](#)

Repositori ini mengimplementasikan pipeline aktuaria terstandarisasi untuk penetapan tarif premi murni (*Pure Premium*) pada produk asuransi kendaraan bermotor (*Motor Third-Party Liability*). Pendekatan yang digunakan membandingkan model klasik dua tahap (*Two-Part GLM: Poisson & Gamma*) dengan model parametrik langsung (*Tweedie Compound Poisson-Gamma Regression*).

---

## 1. Ikhtisar Dataset & Problem Formulation

* **Dataset**: French Motor Third-Party Liability (`freMTPL2freq` & `freMTPL2sev`) via OpenML.
* **Ukuran Sampel**: 677.991 data polis asuransi dengan variabel eksposur waktu, frekuensi klaim, dan total nilai klaim.
* **Fitur Risiko**: `Exposure`, `VehPower`, `VehAge`, `DrivAge`, `BonusMalus`, `VehBrand`, `VehGas`, `Area`, `Density`, dan `Region`.

### Formulasi Aktuaria:
Tingkat premi murni (*Pure Premium*) didefinisikan sebagai ekspektasi total kerugian per unit eksposur:
$$\text{Pure Premium} = \frac{\mathbb{E}[\text{Total Loss}]}{\text{Exposure}} = \mathbb{E}[\text{Frequency}] \times \mathbb{E}[\text{Severity}]$$

1. **Frequency Model**: Poisson Regression dengan target laju klaim ($\lambda$) dan bobot eksposur:
   $$\mathbb{E}[N | X] = \text{Exposure} \times \exp(X\beta_{\text{freq}})$$
2. **Severity Model**: Gamma Regression dengan target besaran biaya rata-rata klaim kondisional ($\mu$) untuk polis berklaim ($N > 0$):
   $$\mathbb{E}[S | X, N > 0] = \exp(X\beta_{\text{sev}})$$
3. **Compound Tweedie Model**: Tweedie Regression ($1 < p < 2$, parameter $p = 1.5$) yang langsung memodelkan distribusi kontinu non-negatif dengan massa probabilitas diskrit di angka nol.

---

## 2. Struktur Repositori

```
insurance-pricing-glm/
├── data/                       # Cache file freMTPL2freq.csv dan freMTPL2sev.csv
├── images/                     # Visualisasi EDA & Evaluasi Model 300 DPI
│   ├── actuarial_risk_eda.png
│   └── lorenz_gini_comparison.png
├── notebook.ipynb              # Master Jupyter Notebook 6-tahap
├── requirements.txt            # Dependensi Python
└── README.md                   # Dokumentasi proyek
```

---

## 3. Hasil Analisis & Evaluasi Model

### Profil Risiko (Exploratory Data Analysis)
Analisis menunjukkan korelasi kuat antara usia pengemudi muda (18-25 tahun) serta nilai koefisien *Bonus-Malus* tinggi dengan peningkatan frekuensi klaim.

![Profil Risiko EDA](images/actuarial_risk_eda.png)

### Evaluasi Diskriminasi Tarif (Lorenz Curve & Gini Index)
Kemampuan model dalam membedakan risiko antar kelompok polis diukur menggunakan kurva Lorenz dan koefisien Gini pada data pengujian terisolasi (*test set*):

![Kurva Lorenz dan Gini](images/lorenz_gini_comparison.png)

* **Two-Part GLM (Poisson $\times$ Gamma)**: Menghasilkan pemisahan risiko yang stabil dengan kemampuan interpretasi aktuaria langsung per komponen frekuensi dan severity.
* **Tweedie Regressor ($p=1.5$)**: Menghasilkan pemodelan langsung satu langkah (*end-to-end*) dengan efisiensi komputasi tinggi.

---

## 4. Cara Menjalankan

1. **Clone repositori dan pasang dependensi**:
   ```bash
   git clone https://github.com/your-username/insurance-pricing-glm.git
   cd insurance-pricing-glm
   pip install -r requirements.txt
   ```

2. **Jalankan Jupyter Notebook**:
   ```bash
   jupyter notebook notebook.ipynb
   ```

---
*Proyek 01 dari Seri Portofolio Data Science Industri Asuransi.*
