import pandas as pd
import pickle

# Load model terbaik (pipeline lengkap) dari file
with open('model/best_random_forest_model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

# Data baru sebagai DataFrame (pastikan kolom sama seperti X saat training)
new_data = pd.DataFrame([{
    'job': 'white-collar',
    'marital': 'married',
    'balance': 15000000000,
    'age_group': 'gen_x',
    'is_having_debt': 2
}])

# Lakukan prediksi
prediction = loaded_model.predict(new_data)

# Jika ingin probabilitas per kelas
proba = loaded_model.predict_proba(new_data)

print(f"🎯 Prediksi Klaster untuk Data Baru: {prediction}")
print(f"🎯 Confidence Score masing - masing Klaster untuk Data Baru: {proba}")