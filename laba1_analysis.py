# -*- coding: utf-8 -*-

import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 11
plt.rcParams["axes.titlesize"] = 13
plt.rcParams["axes.labelsize"] = 11
plt.rcParams["figure.dpi"] = 110


def main():
    show_plots = "--no-show" not in sys.argv

    file_path = "house_price_regression_dataset.csv"
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Файл {file_path} не найден.")

    df_raw = pd.read_csv(file_path)
    print(f"Размерность исходного датасета: {df_raw.shape[0]} строк, {df_raw.shape[1]} столбцов\n")

    col_map = {
        "Square_Footage": "Площадь",
        "Num_Bedrooms": "Спальни",
        "Num_Bathrooms": "Ванные",
        "Year_Built": "Год постройки",
        "Lot_Size": "Участок",
        "Garage_Size": "Гараж",
        "Neighborhood_Quality": "Качество района",
        "House_Price": "Цена дома",
    }
    df = df_raw.rename(columns=col_map)

    print("Первые 5 строк датасета:")
    print(df.head().to_string())

    print("\nПропущенные значения по столбцам:")
    print(df.isnull().sum().to_string())

    duplicates_count = int(df.duplicated().sum())
    if duplicates_count > 0:
        df = df.drop_duplicates().reset_index(drop=True)
        print(f"\nОбнаружено и удалено дубликатов: {duplicates_count}. Новый размер: {df.shape}")
    else:
        print("\nДубликаты не обнаружены.")

    numeric_cols = list(df.columns)
    stats_df = df[numeric_cols].describe().T
    stats_df["Медиана"] = df[numeric_cols].median()
    stats_df["Мода"] = [df[c].mode().iloc[0] for c in numeric_cols]
    stats_df["IQR"] = stats_df["75%"] - stats_df["25%"]
    stats_df["Асимметрия"] = df[numeric_cols].skew()
    stats_df = stats_df.rename(columns={
        "count": "Количество",
        "mean": "Среднее",
        "std": "Станд. откл.",
        "min": "Мин.",
        "25%": "25%",
        "50%": "50%",
        "75%": "75%",
        "max": "Макс.",
    })
    print("\nОписательная статистика числовых признаков:")
    print(stats_df[[
        "Количество", "Среднее", "Станд. откл.", "Медиана", "Мода",
        "IQR", "Мин.", "Макс.", "Асимметрия"
    ]].round(2).to_string())

    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    colors = ["#2b5c8f", "#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b", "#e377c2", "#d62728"]
    for ax, col, color in zip(axes.ravel(), numeric_cols, colors):
        sns.histplot(df[col], kde=True, ax=ax, color=color, bins=20, edgecolor="black")
        ax.axvline(df[col].mean(), color="red", linestyle="--", label=f"Среднее: {df[col].mean():.1f}")
        ax.axvline(df[col].median(), color="green", linestyle="-", label=f"Медиана: {df[col].median():.1f}")
        ax.set_title(f"Распределение: {col}")
        ax.set_xlabel(col)
        ax.set_ylabel("Частота")
        ax.legend(fontsize=8)
    plt.tight_layout()
    if show_plots:
        plt.show()
    plt.close()

    features = [c for c in df.columns if c != "Цена дома"]
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    for ax, col in zip(axes.ravel(), features):
        sns.scatterplot(data=df, x=col, y="Цена дома", ax=ax, alpha=0.55, s=28, color="#2b5c8f")
        sns.regplot(data=df, x=col, y="Цена дома", scatter=False, ax=ax, color="red", ci=None)
        ax.set_title(f"Цена vs {col}")
        ax.set_xlabel(col)
        ax.set_ylabel("Цена дома ($)")
    axes.ravel()[-1].axis("off")
    plt.tight_layout()
    if show_plots:
        plt.show()
    plt.close()

    corr_matrix = df.corr(numeric_only=True)
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, linewidths=0.5)
    plt.title("Тепловая карта корреляций Пирсона")
    plt.tight_layout()
    if show_plots:
        plt.show()
    plt.close()

    price_corr = corr_matrix["Цена дома"].drop("Цена дома").sort_values(ascending=False)
    print("\nКорреляция признаков с целевой переменной (Цена дома):")
    print(price_corr.round(3).to_string())

    X_features = df.drop(columns=["Цена дома"])
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X_features), columns=X_features.columns)

    vif_scaled = pd.DataFrame()
    vif_scaled["Признак"] = X_scaled.columns
    vif_scaled["VIF (Стандартизированные)"] = [
        variance_inflation_factor(X_scaled.values, i) for i in range(X_scaled.shape[1])
    ]
    vif_scaled = vif_scaled.sort_values(by="VIF (Стандартизированные)", ascending=False).reset_index(drop=True)

    print("\nТаблица коэффициентов вздутия дисперсии (VIF):")
    print(vif_scaled.round(3).to_string(index=False))

    plt.figure(figsize=(10, 5))
    bars = plt.barh(
        vif_scaled["Признак"],
        vif_scaled["VIF (Стандартизированные)"],
        color="#3182bd",
        edgecolor="black",
    )
    plt.axvline(5.0, color="red", linestyle="--", linewidth=1.5, label="Критический порог VIF = 5.0")
    plt.title("Оценка мультиколлинеарности: значения VIF предикторов")
    plt.xlabel("VIF")
    for bar in bars:
        val = bar.get_width()
        plt.text(val + 0.03, bar.get_y() + bar.get_height() / 2, f"{val:.3f}", va="center", fontsize=10)
    plt.xlim(0, 6)
    plt.legend()
    plt.tight_layout()
    if show_plots:
        plt.show()
    plt.close()


if __name__ == "__main__":
    main()
