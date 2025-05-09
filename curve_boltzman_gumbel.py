# 🔧 Librerie necessarie
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
from google.colab import files

# 📤 Caricamento del file Excel
uploaded = files.upload()
df = pd.read_excel(next(iter(uploaded)))

# 🎯 Modelli di fitting

# Curva di Gumbel (CDF scalata)
def gumbel_scaled(x, A, mu, beta):
    return A * np.exp(-np.exp(-(x - mu) / beta))

# Curva di Boltzmann (sigmoide)
def boltzmann(x, A1, A2, x0, dx):
    return (A1 - A2) / (1 + np.exp((x - x0) / dx)) + A2

# Estrai i nomi delle colonne
columns = df.columns.tolist()
x = df[columns[0]].values  # Prima colonna = concentrazione

# Loop su ciascun analita (dalla seconda colonna in poi)
for col in columns[1:]:
    y = df[col].values

    # Rimuovi eventuali NaN
    mask = ~np.isnan(x) & ~np.isnan(y)
    x_clean, y_clean = x[mask], y[mask]

    # Fit Gumbel
    try:
        gumbel_params, _ = curve_fit(gumbel_scaled, x_clean, y_clean,
                                     p0=[max(y_clean), np.median(x_clean), 0.1], maxfev=10000)
        y_gumbel = gumbel_scaled(x_clean, *gumbel_params)
        r2_gumbel = r2_score(y_clean, y_gumbel)
    except:
        gumbel_params, r2_gumbel = [np.nan]*3, np.nan

    # Fit Boltzmann
    try:
        boltz_params, _ = curve_fit(boltzmann, x_clean, y_clean,
                                    p0=[max(y_clean), min(y_clean), np.median(x_clean), 0.1], maxfev=10000)
        y_boltz = boltzmann(x_clean, *boltz_params)
        r2_boltz = r2_score(y_clean, y_boltz)
    except:
        boltz_params, r2_boltz = [np.nan]*4, np.nan

    # 🔍 Grafico
    x_fit = np.linspace(min(x_clean), max(x_clean), 300)
    y_gumbel_fit = gumbel_scaled(x_fit, *gumbel_params) if not np.isnan(r2_gumbel) else None
    y_boltz_fit = boltzmann(x_fit, *boltz_params) if not np.isnan(r2_boltz) else None

    plt.figure(figsize=(8, 6))
    plt.scatter(x_clean, y_clean, label="Dati sperimentali", color="black")

    if y_gumbel_fit is not None:
        plt.plot(x_fit, y_gumbel_fit, label=f"Fit Gumbel (R² = {r2_gumbel:.3f})", color="orange")

    if y_boltz_fit is not None:
        plt.plot(x_fit, y_boltz_fit, label=f"Fit Boltzmann (R² = {r2_boltz:.3f})", color="blue")

    plt.title(f"Fit Gumbel vs Boltzmann - {col}")
    plt.xlabel("Concentrazione")
    plt.ylabel("Segnale IMS")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
