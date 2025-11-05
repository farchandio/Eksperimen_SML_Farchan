import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.stats import zscore
import os

def load_and_preprocess(csv_path, output_file=None):
    """
    Memuat, membersihkan, dan melakukan preprocessing pada dataset stroke.
    
    Args:
        csv_path (str): Path ke file CSV mentah.
        output_file (str, optional): Path untuk menyimpan file CSV bersih.

    Returns:
        pandas.DataFrame: DataFrame yang sudah diproses.
    """
    # Load dataset
    df = pd.read_csv(csv_path)

    # --- 1. Handle Missing Values & Anomali ---
    bmi_median = df['bmi'].median()
    df['bmi'].fillna(bmi_median, inplace=True)
    
    smoking_mode = df['smoking_status'].mode()[0]
    df['smoking_status'] = df['smoking_status'].replace('Unknown', smoking_mode)
    
    df = df[df['gender'] != 'Other']

    # --- 2. Drop Duplikat ---
    df.drop_duplicates(inplace=True)

    # --- 3. Encoding Data Kategorikal ---
    df['gender'] = df['gender'].map({'Male': 1, 'Female': 0})
    df['ever_married'] = df['ever_married'].map({'Yes': 1, 'No': 0})
    df['Residence_type'] = df['Residence_type'].map({'Urban': 1, 'Rural': 0})

    df = pd.get_dummies(df, columns=['work_type', 'smoking_status'], drop_first=True)

    # --- 4. Drop Kolom ID ---
    df.drop(columns=['id'], inplace=True)

    # --- 5. Normalisasi & Deteksi Outlier ---
    numeric_cols = ['age', 'avg_glucose_level', 'bmi']

    # 1. Lakukan Normalisasi (Scaling) TERLEBIH DAHULU
    # (Ini agar konsisten dengan alur notebook)
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    # 2. Hitung Z-score SETELAH data di-scale
    # JANGAN GUNAKAN .to_numpy()
    # zscore() pada DataFrame akan mengembalikan DataFrame baru dengan INDEKS YANG SAMA
    z_scores_df = np.abs(zscore(df[numeric_cols].values))

    # 3. Buat mask (filter) dari DataFrame z_scores_df
    # Hasilnya adalah pandas Series dengan INDEKS YANG BENAR
    mask = (z_scores_df < 3).all(axis=1)

    # 4. Terapkan filter ke 'df'. 
    # Ini aman karena indeks 'mask' dan 'df' akan cocok dengan benar.
    df = df[mask]

    # --- 6. Simpan File (Opsional) ---
    if output_file:
        output_dir = os.path.dirname(output_file)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        df.to_csv(output_file, index=False)

    return df


# Bagian if __name__ == "__main__": Anda sudah BENAR dan tidak perlu diubah.
if __name__ == "__main__":
    """
    Blok ini akan dieksekusi saat script dijalankan langsung.
    Ini mengasumsikan struktur folder Kriteria 1:
    Eksperimen_SML_Nama-siswa/
    ├── namadataset_raw/
    │   └── healthcare-dataset-stroke-data.csv
    ├── preprocessing/
    │   └── automate_Nama-siswa.py  <-- (File ini)
    └── Membangun_model/
        └── namadataset_preprocessing/  <-- (Lokasi output)
    """
    
    # Dapatkan path absolut dari script ini
    script_path = os.path.abspath(__file__)
    
    # Dapatkan folder .../preprocessing/
    preprocessing_dir = os.path.dirname(script_path)
    
    # Dapatkan folder root proyek, mis: .../Eksperimen_SML_Farchan/
    repo_root = os.path.dirname(preprocessing_dir)
    
    # Tentukan path input
    input_csv = os.path.join(repo_root, 'namadataset_raw', 'healthcare-dataset-stroke-data.csv')
    
    # Tentukan path output
    output_file_path = os.path.join(repo_root, 'Membangun_model', 'namadataset_preprocessing', 'data_bersih.csv')

    print(f"Loading data from: {input_csv}")
    
    # Jalankan fungsi
    load_and_preprocess(input_csv, output_file_path)
    
    print(f"Preprocessing complete. Clean data saved to: {output_file_path}")