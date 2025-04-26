import pandas as pd
import os

def load_data(filepath):
    """
    Wczytuje dane SMS Spam Collection z pliku tekstowego.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Plik nie został znaleziony: {filepath}")

    df = pd.read_csv(filepath, sep='\t', header=None, names=['label', 'message'])
    return df

def basic_info(df):
    print("Pierwsze 5 wiadomości:")
    print(df.head())
    print("\nStatystyki labeli:")
    print(df['label'].value_counts())
    print("\nInformacje o danych:")
    print(df.info())