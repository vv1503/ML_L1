# -*- coding: utf-8 -*-
"""
Лабораторная работа №1. Часть 2: подготовка данных, регрессия, PCA, сравнение.
Датасет: Home Value Insights (house_price_regression_dataset.csv).
"""

import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 11
plt.rcParams["axes.titlesize"] = 13
plt.rcParams["axes.labelsize"] = 11
plt.rcParams["figure.dpi"] = 110


def label_component(loadings_col):
    """Содержательная подпись ГК по доминирующим факторным нагрузкам (после PCA)."""
    ranked = loadings_col.abs().sort_values(ascending=False)
    top_names = list(ranked.index[:3])
    top_vals = loadings_col.loc[top_names]

    def signed(name):
        return f"{name} ({loadings_col[name]:+.2f})"

    # Именование по смыслу топ-признаков
    if top_names[0] == "Год постройки" and ranked.iloc[0] > 0.6:
        title = "Фактор новизны дома"
    elif set(top_names[:2]) >= {"Площадь", "Участок"} or (
        "Площадь" in top_names[:2] and "Участок" in top_names[:2]
    ):
        title = "Фактор площади дома и участка"
    elif top_names[0] == "Гараж" or (
        "Гараж" in top_names[:2] and "Спальни" in top_names[:2]
    ):
        title = "Гаражно-планировочный профиль"
    elif "Качество района" in top_names[:2] and "Ванные" in top_names[:2]:
        if top_vals["Ванные"] * top_vals["Качество района"] > 0:
            title = "Качество района и комфорт санузлов"
        else:
            title = "Район и гараж vs число ванных"
    elif top_names[0] == "Качество района":
        title = "Район и гараж vs число ванных"
    elif top_names[0] == "Спальни":
        title = "Спальный профиль планировки"
    else:
        title = "Смешанный фактор: " + ", ".join(top_names[:2])

    detail = ", ".join(signed(n) for n in top_names)
    return title, detail


def eval_models(models, X_tr, X_te, y_tr, y_te, kf, feature_names):
    results = {}
    preds = {}
    for name, model in models.items():
        cv_scores = cross_val_score(model, X_tr, y_tr, cv=kf, scoring="r2")
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)
        preds[name] = y_pred
        results[name] = {
            "R2_CV_mean": float(cv_scores.mean()),
            "R2_CV_std": float(cv_scores.std()),
            "R2_Test": float(r2_score(y_te, y_pred)),
            "RMSE_Test": float(np.sqrt(mean_squared_error(y_te, y_pred))),
            "MAPE_Test": float(mean_absolute_percentage_error(y_te, y_pred)),
            "intercept": float(model.intercept_),
            "coefficients": {
                col: float(coef) for col, coef in zip(feature_names, model.coef_)
            },
        }
        print(
            f"{name:<48} | R^2(CV): {cv_scores.mean():.4f} | "
            f"R^2(Тест): {results[name]['R2_Test']:.4f} | "
            f"RMSE: {results[name]['RMSE_Test']:.2f} $ | "
            f"MAPE: {results[name]['MAPE_Test'] * 100:.2f}%"
        )
    return results, preds


def main():
    show_plots = "--no-show" not in sys.argv

    if not os.path.exists("house_price_regression_dataset.csv"):
        raise FileNotFoundError("house_price_regression_dataset.csv не найден.")

    df_raw = pd.read_csv("house_price_regression_dataset.csv").drop_duplicates().reset_index(drop=True)
    col_rename = {
        "Square_Footage": "Площадь",
        "Num_Bedrooms": "Спальни",
        "Num_Bathrooms": "Ванные",
        "Year_Built": "Год постройки",
        "Lot_Size": "Участок",
        "Garage_Size": "Гараж",
        "Neighborhood_Quality": "Качество района",
        "House_Price": "Цена дома",
    }
    df = df_raw.rename(columns=col_rename)

    X = df.drop(columns=["Цена дома"])
    y = df["Цена дома"]
    feature_names = list(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=feature_names)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=feature_names)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    # --- Подбор alpha для Ridge и Lasso ---
    alpha_grid = {
        "alpha": [0.01, 0.1, 1.0, 10.0, 50.0, 100.0, 500.0, 1000.0]
    }

    ridge_search = GridSearchCV(
        Ridge(random_state=42), alpha_grid, cv=kf, scoring="r2", n_jobs=-1
    )
    ridge_search.fit(X_train_scaled, y_train)
    best_ridge_alpha = float(ridge_search.best_params_["alpha"])

    lasso_search = GridSearchCV(
        Lasso(max_iter=20000, random_state=42), alpha_grid, cv=kf, scoring="r2", n_jobs=-1
    )
    lasso_search.fit(X_train_scaled, y_train)
    best_lasso_alpha = float(lasso_search.best_params_["alpha"])

    print("=== Подбор гиперпараметров (GridSearchCV, 5-fold, метрика R^2) ===")
    print(f"Лучший alpha для Ridge: {best_ridge_alpha} (CV R^2 = {ridge_search.best_score_:.4f})")
    print(f"Лучший alpha для Lasso: {best_lasso_alpha} (CV R^2 = {lasso_search.best_score_:.4f})")
    print("Сетка alpha:", alpha_grid["alpha"])

    model_lin = LinearRegression()
    model_ridge = Ridge(alpha=best_ridge_alpha, random_state=42)
    model_lasso = Lasso(alpha=best_lasso_alpha, max_iter=20000, random_state=42)

    models_orig = {
        "Линейная регрессия (OLS)": model_lin,
        f"Гребневая регрессия (Ridge, alpha={best_ridge_alpha:g})": model_ridge,
        f"Лассо регрессия (Lasso, alpha={best_lasso_alpha:g})": model_lasso,
    }

    print("\n=== Обучение моделей на исходных стандартизированных признаках ===")
    results_orig, preds_orig = eval_models(
        models_orig, X_train_scaled, X_test_scaled, y_train, y_test, kf, feature_names
    )

    print(
        "\nКритическое замечание: R^2 ≈ 0.998 и корреляция Площадь–Цена ≈ 0.99 "
        "для реальных цен на жильё почти нереальны. Высокое качество здесь "
        "объясняется синтетической природой данных (цена почти линейно от площади), "
        "а не «силой» самого метода регрессии."
    )

    # --- PCA ---
    pca_full = PCA().fit(X_train_scaled)
    eigenvalues = pca_full.explained_variance_
    exp_var_ratio = pca_full.explained_variance_ratio_
    cum_var = np.cumsum(exp_var_ratio)

    k_kaiser = int(np.sum(eigenvalues > 1))
    k_var85 = int(np.argmax(cum_var >= 0.85) + 1)

    print("\n=== Результаты факторного анализа (PCA) ===")
    pca_df = pd.DataFrame({
        "Главная компонента": [f"ГК {i + 1}" for i in range(len(eigenvalues))],
        "Собственное значение (Lambda)": eigenvalues,
        "Доля дисперсии": exp_var_ratio,
        "Накопленная дисперсия": cum_var,
    })
    print(pca_df.round(4).to_string(index=False))
    print(f"По Кайзеру (λ > 1): {k_kaiser} компонент")
    print(f"По порогу 85% дисперсии: {k_var85} компонент")

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    axes[0].plot(range(1, len(eigenvalues) + 1), eigenvalues, "bo-", linewidth=2, markersize=7)
    axes[0].axhline(1.0, color="red", linestyle="--", linewidth=2, label="Критерий Кайзера (λ = 1)")
    axes[0].set_title("График каменистой осыпи (Scree Plot)")
    axes[0].set_xlabel("Номер главной компоненты")
    axes[0].set_ylabel("Собственное значение")
    axes[0].set_xticks(range(1, len(eigenvalues) + 1))
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(range(1, len(cum_var) + 1), cum_var, "ro-", linewidth=2, markersize=7)
    axes[1].axhline(0.85, color="orange", linestyle="--", label="Порог 85%")
    axes[1].axvline(k_kaiser, color="green", linestyle=":", label=f"Кайзер: {k_kaiser}")
    axes[1].axvline(k_var85, color="gray", linestyle=":", label=f"85%: {k_var85}")
    axes[1].set_title("Кумулятивная объяснённая дисперсия")
    axes[1].set_xlabel("Количество главных компонент")
    axes[1].set_ylabel("Доля объяснённой дисперсии")
    axes[1].set_xticks(range(1, len(cum_var) + 1))
    axes[1].set_ylim(0, 1.05)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    if show_plots:
        plt.show()
    plt.close()

    # Подписи строим из нагрузок ПОСЛЕ PCA (не заранее)
    pca_for_labels = PCA(n_components=k_var85).fit(X_train_scaled)
    loadings_full = pd.DataFrame(
        pca_for_labels.components_.T,
        columns=[f"ГК {i + 1}" for i in range(k_var85)],
        index=feature_names,
    )
    pc_labels = {}
    print("\n=== Подписи ГК, полученные из факторных нагрузок ===")
    print(loadings_full.round(3).to_string())
    for i in range(k_var85):
        title, detail = label_component(loadings_full[f"ГК {i + 1}"])
        pc_labels[i + 1] = title
        print(f"ГК {i + 1} — «{title}»: {detail}")

    plt.figure(figsize=(11, 6))
    sns.heatmap(loadings_full, annot=True, fmt=".2f", cmap="coolwarm", center=0, linewidths=0.5)
    plt.title(f"Факторные нагрузки первых {k_var85} главных компонент")
    plt.ylabel("Исходные признаки")
    plt.xlabel("Главные компоненты")
    plt.tight_layout()
    if show_plots:
        plt.show()
    plt.close()

    # Сравнение 3 ГК (Кайзер) и 6 ГК (85%)
    results_pca_all = {}
    for k in sorted({k_kaiser, k_var85}):
        pca_k = PCA(n_components=k).fit(X_train_scaled)
        X_tr_k = pca_k.transform(X_train_scaled)
        X_te_k = pca_k.transform(X_test_scaled)
        names_k = [f"ГК {i + 1}: {pc_labels.get(i + 1, '—')}" for i in range(k)]

        models_k = {
            f"Линейная (PCA, {k} ГК)": LinearRegression(),
            f"Ridge (PCA, {k} ГК, alpha={best_ridge_alpha:g})": Ridge(
                alpha=best_ridge_alpha, random_state=42
            ),
            f"Lasso (PCA, {k} ГК, alpha={best_lasso_alpha:g})": Lasso(
                alpha=best_lasso_alpha, max_iter=20000, random_state=42
            ),
        }
        print(f"\n=== Модели на {k} главных компонентах "
              f"({'Кайзер' if k == k_kaiser else 'порог 85%'}) ===")
        res_k, _ = eval_models(models_k, X_tr_k, X_te_k, y_train, y_test, kf, names_k)
        results_pca_all[k] = res_k

    # Сводная таблица
    summary_rows = []
    for name, res in results_orig.items():
        summary_rows.append({
            "Модель": name,
            "Пространство признаков": f"Исходные стандартизированные ({len(feature_names)})",
            "R^2 (Кросс-вал)": res["R2_CV_mean"],
            "R^2 (Тест)": res["R2_Test"],
            "RMSE (Тест, $)": res["RMSE_Test"],
            "MAPE (Тест, %)": res["MAPE_Test"] * 100,
        })
    for k, res_dict in results_pca_all.items():
        space = f"Главные компоненты ({k}, {'Кайзер' if k == k_kaiser else '85%'})"
        for name, res in res_dict.items():
            summary_rows.append({
                "Модель": name,
                "Пространство признаков": space,
                "R^2 (Кросс-вал)": res["R2_CV_mean"],
                "R^2 (Тест)": res["R2_Test"],
                "RMSE (Тест, $)": res["RMSE_Test"],
                "MAPE (Тест, %)": res["MAPE_Test"] * 100,
            })

    summary_df = pd.DataFrame(summary_rows).sort_values(by="R^2 (Тест)", ascending=False).reset_index(drop=True)
    print("\n=== Итоговая сравнительная таблица всех моделей ===")
    print(summary_df.round(4).to_string(index=False))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    sns.barplot(
        data=summary_df, x="Модель", y="R^2 (Тест)", ax=axes[0],
        hue="Модель", palette="Blues_r", edgecolor="black", legend=False,
    )
    axes[0].set_title("Сравнение R^2 (Тест)")
    axes[0].set_xlabel("")
    axes[0].tick_params(axis="x", rotation=40)

    sns.barplot(
        data=summary_df, x="Модель", y="RMSE (Тест, $)", ax=axes[1],
        hue="Модель", palette="Reds_r", edgecolor="black", legend=False,
    )
    axes[1].set_title("Сравнение RMSE ($)")
    axes[1].set_xlabel("")
    axes[1].tick_params(axis="x", rotation=40)

    sns.barplot(
        data=summary_df, x="Модель", y="MAPE (Тест, %)", ax=axes[2],
        hue="Модель", palette="Greens_r", edgecolor="black", legend=False,
    )
    axes[2].set_title("Сравнение MAPE (%)")
    axes[2].set_xlabel("")
    axes[2].tick_params(axis="x", rotation=40)
    plt.tight_layout()
    if show_plots:
        plt.show()
    plt.close()

    weights_payload = {
        "датасет": "house_price_regression_dataset.csv",
        "обучающая_выборка": int(X_train.shape[0]),
        "тестовая_выборка": int(X_test.shape[0]),
        "подбор_гиперпараметров": {
            "сетка_alpha": alpha_grid["alpha"],
            "лучший_alpha_Ridge": best_ridge_alpha,
            "лучший_alpha_Lasso": best_lasso_alpha,
            "CV_R2_Ridge": float(ridge_search.best_score_),
            "CV_R2_Lasso": float(lasso_search.best_score_),
        },
        "модели_на_исходных_признаках": results_orig,
        "факторный_анализ_pca": {
            "компонент_по_Кайзеру": k_kaiser,
            "компонент_по_порогу_85": k_var85,
            "доля_дисперсии_Кайзер": float(cum_var[k_kaiser - 1]),
            "доля_дисперсии_85": float(cum_var[k_var85 - 1]),
            "подписи_компонент_из_нагрузок": {f"ГК_{i}": pc_labels[i] for i in pc_labels},
            "собственные_значения": [float(e) for e in eigenvalues],
            "факторные_нагрузки": {
                col: {f"ГК_{i + 1}": float(loadings_full.loc[col, f"ГК {i + 1}"]) for i in range(k_var85)}
                for col in feature_names
            },
        },
        "модели_на_главных_компонентах": {
            f"{k}_ГК": results_pca_all[k] for k in results_pca_all
        },
    }

    with open("laba1_model_weights.json", "w", encoding="utf-8") as f:
        json.dump(weights_payload, f, ensure_ascii=False, indent=2)

    weights_csv_df = pd.DataFrame({
        "Признак": feature_names,
        "Линейная_регрессия_OLS": model_lin.coef_,
        "Гребневая_регрессия_Ridge": model_ridge.coef_,
        "Лассо_регрессия_Lasso": model_lasso.coef_,
    })
    weights_csv_df.to_csv("laba1_weights.csv", index=False)

    print("\n=== Таблица весов моделей на стандартизированных признаках ===")
    print(weights_csv_df.round(3).to_string(index=False))
    print(f"Свободный член (Intercept) линейной регрессии: {model_lin.intercept_:.2f} $")
    print("Файлы сохранены: laba1_model_weights.json, laba1_weights.csv")


if __name__ == "__main__":
    main()
