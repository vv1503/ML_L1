import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, KFold, cross_val_score
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


PC_LABELS = {
    1: "Гаражно-планировочный профиль",
    2: "Фактор площади дома и участка",
    3: "Качество района и комфорт санузлов",
    4: "Фактор новизны дома",
    5: "Район и гараж vs число ванных",
    6: "Спальный профиль планировки",
}


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

    model_lin = LinearRegression()
    model_ridge = Ridge(alpha=1.0, random_state=42)
    model_lasso = Lasso(alpha=1.0, max_iter=10000, random_state=42)

    models_orig = {
        "Линейная регрессия (OLS)": model_lin,
        "Гребневая регрессия (Ridge, alpha=1)": model_ridge,
        "Лассо регрессия (Lasso, alpha=1)": model_lasso,
    }

    results_orig = {}
    preds_orig = {}

    print("=== Обучение моделей на исходных стандартизированных признаках ===")
    for name, model in models_orig.items():
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=kf, scoring="r2")
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        preds_orig[name] = y_pred

        results_orig[name] = {
            "R2_CV_mean": float(cv_scores.mean()),
            "R2_CV_std": float(cv_scores.std()),
            "R2_Test": float(r2_score(y_test, y_pred)),
            "RMSE_Test": float(np.sqrt(mean_squared_error(y_test, y_pred))),
            "MAPE_Test": float(mean_absolute_percentage_error(y_test, y_pred)),
            "intercept": float(model.intercept_),
            "coefficients": {col: float(coef) for col, coef in zip(feature_names, model.coef_)},
        }
        print(
            f"{name:<40} | R^2(CV): {cv_scores.mean():.4f} | "
            f"R^2(Тест): {results_orig[name]['R2_Test']:.4f} | "
            f"RMSE: {results_orig[name]['RMSE_Test']:.2f} $ | "
            f"MAPE: {results_orig[name]['MAPE_Test'] * 100:.2f}%"
        )

    pca_full = PCA().fit(X_train_scaled)
    eigenvalues = pca_full.explained_variance_
    exp_var_ratio = pca_full.explained_variance_ratio_
    cum_var = np.cumsum(exp_var_ratio)

    pca_df = pd.DataFrame({
        "Главная компонента": [f"ГК {i + 1}" for i in range(len(eigenvalues))],
        "Содержательная подпись": [PC_LABELS.get(i + 1, "—") for i in range(len(eigenvalues))],
        "Собственное значение (Lambda)": eigenvalues,
        "Доля дисперсии": exp_var_ratio,
        "Накопленная дисперсия": cum_var,
    })
    print("\n=== Результаты факторного анализа (PCA) ===")
    print(pca_df.round(4).to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    axes[0].plot(range(1, len(eigenvalues) + 1), eigenvalues, "bo-", linewidth=2, markersize=7)
    axes[0].axhline(1.0, color="red", linestyle="--", linewidth=2, label="Критерий Кайзера (Lambda = 1.0)")
    axes[0].set_title("График каменистой осыпи (Scree Plot)")
    axes[0].set_xlabel("Номер главной компоненты")
    axes[0].set_ylabel("Собственное значение")
    axes[0].set_xticks(range(1, len(eigenvalues) + 1))
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(range(1, len(cum_var) + 1), cum_var, "ro-", linewidth=2, markersize=7, label="Накопленная дисперсия")
    axes[1].axhline(0.85, color="orange", linestyle="--", label="Порог 85% дисперсии")
    axes[1].set_title("Кумулятивная объясненная дисперсия")
    axes[1].set_xlabel("Количество главных компонент")
    axes[1].set_ylabel("Доля объясненной дисперсии")
    axes[1].set_xticks(range(1, len(cum_var) + 1))
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    if show_plots:
        plt.show()
    plt.close()

    k_opt = int(np.argmax(cum_var >= 0.85) + 1)
    pca_opt = PCA(n_components=k_opt).fit(X_train_scaled)
    pca_feature_names = [f"ГК {i + 1}: {PC_LABELS[i + 1]}" for i in range(k_opt)]
    loadings = pd.DataFrame(
        pca_opt.components_.T,
        columns=[f"ГК {i + 1}" for i in range(k_opt)],
        index=feature_names,
    )

    print("\n=== Факторные нагрузки (как читать каждую ГК) ===")
    print(loadings.round(3).to_string())
    for i in range(k_opt):
        col = f"ГК {i + 1}"
        top = loadings[col].abs().sort_values(ascending=False).head(3)
        signed = loadings[col].loc[top.index]
        parts = [f"{name} ({val:+.2f})" for name, val in signed.items()]
        print(f"{col} — «{PC_LABELS[i + 1]}»: {', '.join(parts)}")

    plt.figure(figsize=(11, 6))
    sns.heatmap(loadings, annot=True, fmt=".2f", cmap="coolwarm", center=0, linewidths=0.5)
    plt.title(f"Факторные нагрузки первых {k_opt} главных компонент")
    plt.ylabel("Исходные признаки")
    plt.xlabel("Главные компоненты")
    plt.tight_layout()
    if show_plots:
        plt.show()
    plt.close()

    X_train_pca = pca_opt.transform(X_train_scaled)
    X_test_pca = pca_opt.transform(X_test_scaled)

    model_lin_pca = LinearRegression()
    model_ridge_pca = Ridge(alpha=1.0, random_state=42)
    model_lasso_pca = Lasso(alpha=1.0, max_iter=10000, random_state=42)

    models_pca = {
        "Линейная регрессия (на ГК)": model_lin_pca,
        "Гребневая регрессия (Ridge на ГК, alpha=1)": model_ridge_pca,
        "Лассо регрессия (Lasso на ГК, alpha=1)": model_lasso_pca,
    }

    results_pca = {}
    preds_pca = {}

    print("\n=== Обучение моделей регрессии на главных компонентах ===")
    for name, model in models_pca.items():
        cv_scores = cross_val_score(model, X_train_pca, y_train, cv=kf, scoring="r2")
        model.fit(X_train_pca, y_train)
        y_pred = model.predict(X_test_pca)
        preds_pca[name] = y_pred

        results_pca[name] = {
            "R2_CV_mean": float(cv_scores.mean()),
            "R2_CV_std": float(cv_scores.std()),
            "R2_Test": float(r2_score(y_test, y_pred)),
            "RMSE_Test": float(np.sqrt(mean_squared_error(y_test, y_pred))),
            "MAPE_Test": float(mean_absolute_percentage_error(y_test, y_pred)),
            "intercept": float(model.intercept_),
            "coefficients": {
                col: float(coef) for col, coef in zip(pca_feature_names, model.coef_)
            },
        }
        print(
            f"{name:<44} | R^2(CV): {cv_scores.mean():.4f} | "
            f"R^2(Тест): {results_pca[name]['R2_Test']:.4f} | "
            f"RMSE: {results_pca[name]['RMSE_Test']:.2f} $ | "
            f"MAPE: {results_pca[name]['MAPE_Test'] * 100:.2f}%"
        )

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
    for name, res in results_pca.items():
        summary_rows.append({
            "Модель": name,
            "Пространство признаков": f"Главные компоненты ({k_opt})",
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
    axes[0].set_title("Сравнение коэффициента детерминации R^2 (Тест)")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("R^2")
    axes[0].tick_params(axis="x", rotation=35)

    sns.barplot(
        data=summary_df, x="Модель", y="RMSE (Тест, $)", ax=axes[1],
        hue="Модель", palette="Reds_r", edgecolor="black", legend=False,
    )
    axes[1].set_title("Сравнение среднеквадратичной ошибки RMSE ($)")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("RMSE ($)")
    axes[1].tick_params(axis="x", rotation=35)

    sns.barplot(
        data=summary_df, x="Модель", y="MAPE (Тест, %)", ax=axes[2],
        hue="Модель", palette="Greens_r", edgecolor="black", legend=False,
    )
    axes[2].set_title("Сравнение относительной ошибки MAPE (%)")
    axes[2].set_xlabel("")
    axes[2].set_ylabel("MAPE (%)")
    axes[2].tick_params(axis="x", rotation=35)

    plt.tight_layout()
    if show_plots:
        plt.show()
    plt.close()

    weights_payload = {
        "датасет": "house_price_regression_dataset.csv",
        "обучающая_выборка": int(X_train.shape[0]),
        "тестовая_выборка": int(X_test.shape[0]),
        "модели_на_исходных_признаках": results_orig,
        "факторный_анализ_pca": {
            "количество_компонент": k_opt,
            "доля_накопленной_дисперсии": float(cum_var[k_opt - 1]),
            "подписи_компонент": {f"ГК_{i + 1}": PC_LABELS[i + 1] for i in range(k_opt)},
            "собственные_значения": [float(e) for e in eigenvalues],
            "факторные_нагрузки": {
                col: {f"ГК_{i + 1}": float(loadings.loc[col, f"ГК {i + 1}"]) for i in range(k_opt)}
                for col in feature_names
            },
        },
        "модели_на_главных_компонентах": results_pca,
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
