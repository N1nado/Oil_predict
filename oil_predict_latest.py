import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GridSearchCV
from catboost import CatBoostRegressor

# === ЗАГРУЗКА И НАЧАЛЬНАЯ ОЧИСТКА

filename = 'Нефть_.xlsx'
df = pd.read_excel(filename, sheet_name='SubMaster')

# Выводим названия числовых столбцов
numeric_columns = df.select_dtypes(include=['number']).columns
print("Названия числовых столбцов:")
print(numeric_columns.tolist())
print()

# Количество строк в каждом столбце
print("Количество всех строк в каждом столбце:")
print(len(df))
print()

# Проверка на дубликаты
print("Количество полностью одинаковых строк (дубликатов):")
duplicates_count = df.duplicated().sum()
print(duplicates_count)
print()

# Избавляемся от дубликатов
if duplicates_count > 0:
    df = df.drop_duplicates()
    print(f"Удалено дубликатов: {duplicates_count}")
    print(f"Осталось строк: {len(df)}")
else:
    print("Дубликатов не найдено, удаление не требуется.")
print()

# Количество значений NaN
print("Количество значений NaN в каждом столбце:")
print(df.isna().sum())
print()

# Количество значений 0.00
print("Количество значений 0.00 в каждом столбце до замены:")
zeros_before = (df == 0.00).sum()
print(zeros_before)
print()

# Замена нулей
print("Замена значений 0.00 на средние значения по столбцам")
for column in numeric_columns:
    zero_mask = df[column] == 0.00
    zero_count = zero_mask.sum()

    if zero_count > 0:
        mean_value = df[column].replace(0.00, np.nan).mean()
        df.loc[zero_mask, column] = mean_value
        print(f"Столбец '{column}': заменено {zero_count} значений 0.00")
print()

# Количество нулей после замены
print("Количество значений 0.00 в каждом столбце ПОСЛЕ замены:")
print((df == 0.00).sum())
print()

# Уникальные строки
print("Количество полностью уникальных строк после очистки:")
print(len(df.drop_duplicates()))
print()

# Количество уникальных значений
print("Количество уникальных значений для каждого столбца:")
print(df.nunique())
print()

# Статистика
print("Минимальные значения:")
print(df.min()); print()

print("Максимальные значения:")
print(df.max()); print()

print("Средние значения:")
print(df.mean()); print()

print("Медианы:")
print(df.median()); print()

print("Стандартные отклонения:")
print(df.std()); print()

# Гистограммы
for column in numeric_columns:
    plt.figure(figsize=(10, 6))
    plt.hist(df[column].dropna(), bins=15, color='skyblue', edgecolor='black', alpha=0.7)
    plt.title(f'Гистограмма: {column}')
    plt.xlabel(column)
    plt.ylabel('Количество значений')
    plt.tight_layout()
    plt.show()

# МАТРИЦА КОРРЕЛЯЦИЙ до очистки
df_numeric = df.select_dtypes(include=['number'])
corr_matrix = df_numeric.corr()

fig, ax = plt.subplots(figsize=(20, 15))
im = ax.imshow(corr_matrix.values, cmap='coolwarm', vmin=-1, vmax=1)

ax.set_xticks(range(len(corr_matrix.columns)))
ax.set_yticks(range(len(corr_matrix.columns)))
ax.set_xticklabels(corr_matrix.columns, rotation=90)
ax.set_yticklabels(corr_matrix.columns)

for i in range(len(corr_matrix.columns)):
    for j in range(len(corr_matrix.columns)):
        color = "white" if abs(corr_matrix.iloc[i, j]) > 0.5 else "black"
        ax.text(j, i, f'{corr_matrix.iloc[i, j]:.3f}',
                ha='center', va='center', color=color, fontsize=9)

plt.colorbar(im, ax=ax, shrink=0.8)
plt.title('Матрица корреляций Пирсона (до очистки)', fontsize=16)
plt.tight_layout()
plt.show()

# УДАЛЕНИЕ ПРИЗНАКОВ-УТЕЧЕК
TARGET = 'Дебит нефти скважины, м3/сут'

leakage = [
    'Дебит жидкости скважины, м3/сут',
    'Дебит воды скважины, м3/сут',
    'Депрессия (разница пластового и забойного давлений), атм'
]

numeric_columns = df.select_dtypes(include=['number']).columns
X = df[numeric_columns].drop(columns=[TARGET], errors='ignore')
y = df[TARGET]

existing_leaks = [c for c in leakage if c in X.columns]
X = X.drop(columns=existing_leaks)

print("\nУдалены признаки-утечки:", existing_leaks)

# ОЧИСТКА ВЫБРОСОВ ПО IQR
def iqr_clean(s):
    Q1 = s.quantile(0.25)
    Q3 = s.quantile(0.75)
    IQR = Q3 - Q1
    low = Q1 - 1.5 * IQR
    high = Q3 + 1.5 * IQR
    return np.clip(s, low, high)

print("\nОчистка выбросов по IQR...")
for col in X.columns:
    X[col] = iqr_clean(X[col])

y = iqr_clean(y)

# BOXPLOT после удаления выбросов IQR
print("\nBoxPlot после очистки выбросов (IQR):")

for col in X.columns:
    plt.figure(figsize=(8, 5))
    plt.boxplot(X[col], vert=True, patch_artist=True)
    plt.title(f'BoxPlot после IQR очистки: {col}')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

# BoxPlot целевой переменной
plt.figure(figsize=(8, 5))
plt.boxplot(y, vert=True, patch_artist=True, boxprops=dict(facecolor='lightblue'))
plt.title('BoxPlot целевой переменной после IQR очистки')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# УДАЛЕНИЕ СИЛЬНЫХ КОРРЕЛЯЦИЙ > 0.95
corr = X.corr().abs()
to_drop = set()

for i in range(len(corr.columns)):
    for j in range(i):
        if corr.iloc[i, j] > 0.95:
            a, b = corr.columns[i], corr.columns[j]
            ca, cb = abs(X[a].corr(y)), abs(X[b].corr(y))
            to_drop.add(b if ca >= cb else a)

X = X.drop(columns=list(to_drop))
print("\nУдалено по корреляции:", to_drop)

# VIF
def compute_vif(df_):
    df_filled = df_.fillna(df_.median())
    vif_list = []
    for i in range(df_.shape[1]):
        try:
            vif_val = variance_inflation_factor(df_filled.values, i)
        except Exception:
            vif_val = np.nan
        vif_list.append(vif_val)
    return pd.DataFrame({"feature": df_.columns, "VIF": vif_list}).sort_values("VIF", ascending=False)

vif_df = compute_vif(X)
removed_vif = []

while vif_df["VIF"].max() > 10:
    feature_to_remove = vif_df.iloc[0]["feature"]
    removed_vif.append(feature_to_remove)
    X = X.drop(columns=[feature_to_remove])
    vif_df = compute_vif(X)

print("\nУдалено по VIF:", removed_vif)

# МАТРИЦА КОРРЕЛЯЦИЙ после очистки данных
df_clean = X.copy()
df_clean[TARGET] = y

corr_clean = df_clean.corr()

fig, ax = plt.subplots(figsize=(20, 15))
im = ax.imshow(corr_clean.values, cmap='coolwarm', vmin=-1, vmax=1)

ax.set_xticks(range(len(corr_clean.columns)))
ax.set_yticks(range(len(corr_clean.columns)))
ax.set_xticklabels(corr_clean.columns, rotation=90)
ax.set_yticklabels(corr_clean.columns)

for i in range(len(corr_clean.columns)):
    for j in range(len(corr_clean.columns)):
        color = "white" if abs(corr_clean.iloc[i, j]) > 0.5 else "black"
        ax.text(j, i, f'{corr_clean.iloc[i, j]:.3f}',
                ha='center', va='center', color=color, fontsize=9)

plt.colorbar(im, ax=ax, shrink=0.8)
plt.title('Матрица корреляций Пирсона (после очистки)', fontsize=16)
plt.tight_layout()
plt.show()

# POWER TRANSFORM + STANDARD SCALER
skew_cols = X.skew().abs()
skew_cols = skew_cols[skew_cols > 1].index.tolist()

if len(skew_cols) > 0:
    print("\nPowerTransform на признаках:", skew_cols)
    pt = PowerTransformer()
    X[skew_cols] = pt.fit_transform(X[skew_cols])

X = X.fillna(X.median())

scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# ЛОГАРИФМИРОВАНИЕ ЦЕЛЕВОЙ ПЕРЕМЕННОЙ
if abs(y.skew()) > 1:
    print("\nЦель сильно скошена → log1p применён")
    y_log = np.log1p(y)
    log_target = True
else:
    y_log = y
    log_target = False


# TRAIN/TEST
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_log, test_size=0.2, random_state=42
)

# ЭТАП 3 — Обучение базовых моделей
print("\n--- ЭТАП 3: Обучение базовых моделей ---\n")

models = {
    "Linear Regression": LinearRegression(),
    "RandomForest_default": RandomForestRegressor(random_state=42),
    "GradientBoosting_default": GradientBoostingRegressor(random_state=42),
    "CatBoost_default": CatBoostRegressor(random_seed=42, verbose=0)
}

results = {}

for name, model in models.items():
    print(f"\nМодель: {name}")

    # Обучение
    model.fit(X_train, y_train)

    # Предсказания
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Обратное преобразование, если был логарифм
    if log_target:
        y_train_real = np.expm1(y_train)
        y_test_real = np.expm1(y_test)
        y_train_pred_real = np.expm1(y_train_pred)
        y_test_pred_real = np.expm1(y_test_pred)
    else:
        y_train_real = y_train
        y_test_real = y_test
        y_train_pred_real = y_train_pred
        y_test_pred_real = y_test_pred

    # Метрики
    r2_train = r2_score(y_train_real, y_train_pred_real)
    r2_test = r2_score(y_test_real, y_test_pred_real)
    mae_test = mean_absolute_error(y_test_real, y_test_pred_real)
    mse_test = mean_squared_error(y_test_real, y_test_pred_real)

    print(f"R2_train = {r2_train:.4f}")
    print(f"R2_test  = {r2_test:.4f}")
    print(f"MAE_test = {mae_test:.4f}")
    print(f"MSE_test = {mse_test:.4f}")

    results[name] = {
        "model": model,
        "r2_train": r2_train,
        "r2_test": r2_test,
        "mae_test": mae_test,
        "mse_test": mse_test
    }

print("\nЭТАП 3 завершён\n")

# ЭТАП 4 — Подбор гиперпараметров
print("\nЭТАП 4: Подбор гиперпараметров\n")

# ---------- RandomForest ----------
rf = RandomForestRegressor(random_state=42)
param_grid_rf = {
    'n_estimators': [150, 250, 350],
    'max_depth': [5, 7, 9, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_rf = GridSearchCV(
    estimator=rf,
    param_grid=param_grid_rf,
    scoring='r2',
    cv=5,
    n_jobs=-1,
    verbose=1
)

print("Подбор параметров для RandomForest...")
grid_rf.fit(X_train, y_train)

print("\nЛучшие параметры для RandomForest:")
print(grid_rf.best_params_)

best_rf = grid_rf.best_estimator_

# ---------- CatBoost ----------
cb = CatBoostRegressor(random_seed=42, verbose=0)

param_grid_cb = {
    'depth': [4, 5, 6, 8],
    'learning_rate': [0.01, 0.03, 0.05],
    'iterations': [300, 500, 800],
    'l2_leaf_reg': [1, 3, 5]
}

grid_cb = GridSearchCV(
    estimator=cb,
    param_grid=param_grid_cb,
    scoring='r2',
    cv=3,
    n_jobs=-1,
    verbose=1
)

print("\nПодбор параметров для CatBoost...")
grid_cb.fit(X_train, y_train)

print("\nЛучшие параметры CatBoost:")
print(grid_cb.best_params_)

best_cb = grid_cb.best_estimator_

# Предсказания CatBoost
y_pred_cb = best_cb.predict(X_test)

if log_target:
    y_pred_cb_real = np.expm1(y_pred_cb)
    y_test_real = np.expm1(y_test)
else:
    y_pred_cb_real = y_pred_cb
    y_test_real = y_test

print("\nCatBoost результаты (tuned):")
print("R2 =", r2_score(y_test_real, y_pred_cb_real))
print("MAE =", mean_absolute_error(y_test_real, y_pred_cb_real))
print("MSE =", mean_squared_error(y_test_real, y_pred_cb_real))

# ЭТАП 5 — Важность признаков (CatBoost + Permutation Importance)
print("\nЭТАП 5 — Feature Importance\n")

final_features = X.columns.tolist()

# CatBoost Feature Importance
fi = best_cb.get_feature_importance()

print("\nCatBoost Feature Importance:")
for name, val in sorted(zip(final_features, fi), key=lambda x: -x[1]):
    print(f"{name:45s} = {val:.4f}")

plt.figure(figsize=(10, 6))
plt.barh(final_features, fi)
plt.title("CatBoost Feature Importance")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# Permutation Importance
print("\nPermutation Importance (CatBoost)...")

perm = permutation_importance(
    best_cb, X_test, y_test,
    n_repeats=20,
    random_state=42
)

perm_imp = perm.importances_mean
perm_std = perm.importances_std
sorted_idx = np.argsort(perm_imp)[::-1]

for idx in sorted_idx:
    print(f"{final_features[idx]:45s} = {perm_imp[idx]:.4f}")

plt.figure(figsize=(10, 6))
plt.barh(np.array(final_features)[sorted_idx], perm_imp[sorted_idx], xerr=perm_std[sorted_idx])
plt.title("Permutation Importance (CatBoost)")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# ЭТАП 6 — Финальное сравнение моделей
print("\nЭТАП 6 — Финальное сравнение моделей\n")

final_models = {
    "Linear Regression": results["Linear Regression"]["model"],
    "RandomForest_default": results["RandomForest_default"]["model"],
    "GradientBoosting_default": results["GradientBoosting_default"]["model"],
    "CatBoost_default": results["CatBoost_default"]["model"],
    "RandomForest_tuned": best_rf,
    "CatBoost_tuned": best_cb
}

summary = []

for name, model in final_models.items():
    print(f"Оцениваем модель: {name}")

    y_pred = model.predict(X_test)

    if log_target:
        y_pred_real = np.expm1(y_pred)
        y_test_real = np.expm1(y_test)
    else:
        y_pred_real = y_pred
        y_test_real = y_test

    r2 = r2_score(y_test_real, y_pred_real)
    mae = mean_absolute_error(y_test_real, y_pred_real)
    mse = mean_squared_error(y_test_real, y_pred_real)

    summary.append([name, r2, mae, mse])


summary_df = pd.DataFrame(summary, columns=["Модель", "R2", "MAE", "MSE"])
summary_df = summary_df.sort_values(by="R2", ascending=False).reset_index(drop=True)

print("\nИтоговое сравнение моделей:")
print(summary_df)

plt.figure(figsize=(10, 5))
plt.bar(summary_df["Модель"], summary_df["R2"], color="teal")
plt.xticks(rotation=45, ha="right")
plt.ylabel("R2 score")
plt.title("Сравнение моделей")
plt.tight_layout()
plt.show()

best_model = summary_df.iloc[0]
print("\nЛУЧШАЯ МОДЕЛЬ:")
print(best_model)

print("\nЭТАП 6 завершён\n")

# ГРАФИКИ ФАКТ vs ПРОГНОЗ

def plot_fact_vs_pred(y_true, y_pred, title):
    plt.figure(figsize=(7, 7))
    plt.scatter(y_true, y_pred, alpha=0.6)
    plt.plot([y_true.min(), y_true.max()],
             [y_true.min(), y_true.max()],
             color='red', linestyle='--', linewidth=2)
    plt.xlabel("Фактические значения")
    plt.ylabel("Прогноз модели")
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

print("\n=== ГРАФИКИ ФАКТ vs ПРОГНОЗ ===")

# CatBoost (лучшая модель)
y_pred_cb = best_cb.predict(X_test)
y_pred_cb_real = np.expm1(y_pred_cb) if log_target else y_pred_cb
y_test_real = np.expm1(y_test) if log_target else y_test
plot_fact_vs_pred(y_test_real, y_pred_cb_real, "CatBoost_tuned: факт vs прогноз")

# RandomForest
y_pred_rf = best_rf.predict(X_test)
y_pred_rf_real = np.expm1(y_pred_rf) if log_target else y_pred_rf
plot_fact_vs_pred(y_test_real, y_pred_rf_real, "RandomForest_tuned: факт vs прогноз")

# GradientBoosting_default
gb_model = results["GradientBoosting_default"]["model"]
y_pred_gb = gb_model.predict(X_test)
y_pred_gb_real = np.expm1(y_pred_gb) if log_target else y_pred_gb
plot_fact_vs_pred(y_test_real, y_pred_gb_real, "GradientBoosting: факт vs прогноз")