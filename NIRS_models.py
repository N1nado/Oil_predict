# Подключение библиотек
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor

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

# Количество значений NaN в каждом столбце
print("Количество значений NaN в каждом столбце:")
print(df.isna().sum())
print()

# Количество значений 0.00 в каждом столбце до замены
print("Количество значений 0.00 в каждом столбце до замены:")
zeros_before = (df == 0.00).sum()
print(zeros_before)
print()

# Замена значений 0.00 на средние значения по столбцам
print("Замена значений 0.00 на средние значения по столбцам")
for column in numeric_columns:
    # Проверяем, есть ли нулевые значения в столбце
    zero_mask = df[column] == 0.00
    zero_count = zero_mask.sum()
    
    if zero_count > 0:
        # Вычисляем среднее значение столбца, исключая нули и NaN
        mean_value = df[column].replace(0.00, np.nan).mean()
        
        # Заменяем нули на среднее значение
        df.loc[zero_mask, column] = mean_value
        
        print(f"Столбец '{column}': заменено {zero_count} значений 0.00 на среднее {mean_value:.6f}")

print()

# Количество значений 0.00 в каждом столбце после замены
print("Количество значений 0.00 в каждом столбце ПОСЛЕ замены:")
zeros_after = (df == 0.00).sum()
print(zeros_after)
print()

# Количество полностью уникальных строк после очистки
print("Количество полностью уникальных строк после очистки:")
unique_rows_count = len(df.drop_duplicates())
print(unique_rows_count)

# Количество уникальных значений для каждого столбца
print("Количество уникальных значений для каждого столбца:")
print(df.nunique())
print()

# Статистика по столбцам
print("Минимальные значения:")
print(df.min())
print()

print("Максимальные значения:")
print(df.max())
print()

print("Средние значения:")
print(df.mean())
print()

print("Медианы:")
print(df.median())
print()

print("Стандартные отклонения (среднее квадратичное отклонение):")
print(df.std())

# Строим гистограммы для всех числовых столбцов
for column in numeric_columns:
    plt.figure(figsize=(10, 6))
    plt.hist(df[column].dropna(), bins=15, color='skyblue', edgecolor='black', alpha=0.7)
    plt.title(f'Гистограмма: {column}')
    plt.xlabel(column)
    plt.ylabel('Количество значений')
    plt.tight_layout()
    plt.show()
    
# Анализ мультиколлинеарности для числовых переменных
# Матрица корреляций Пирсона
# Создаем DataFrame только с числовыми столбцами
df_numeric = df.select_dtypes(include=['number'])

# Матрица корреляций Пирсона
corr_matrix = df_numeric.corr()
    
# Визуализация матрицы корреляций
fig, ax = plt.subplots(figsize=(20, 15))
    
# Создаем heatmap
im = ax.imshow(corr_matrix.values, cmap='coolwarm', vmin=-1, vmax=1)
    
# Добавляем подписи
ax.set_xticks(range(len(corr_matrix.columns)))
ax.set_yticks(range(len(corr_matrix.columns)))
ax.set_xticklabels(corr_matrix.columns, rotation = 90, ha = 'right')
ax.set_yticklabels(corr_matrix.columns)
    
# Добавляем значения в ячейки
for i in range(len(corr_matrix.columns)):
    for j in range(len(corr_matrix.columns)):
        color = "white" if abs(corr_matrix.iloc[i, j]) > 0.5 else "black"
        text = ax.text(j, i, f'{corr_matrix.iloc[i, j]:.3f}',
                      ha="center", va="center", color=color, fontsize=12, fontweight='bold')
    
# Добавляем цветовую шкалу
cbar = plt.colorbar(im, ax=ax, shrink=0.8)
cbar.set_label('Коэффициент корреляции', rotation=270, labelpad=20)
 
plt.title('Матрица корреляций Пирсона числовых переменных', fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.show()

# Box plot для каждого числового столбца для анализа выбросов
print("Анализ выбросов с помощью Box Plot:")
for column in numeric_columns:
    plt.figure(figsize=(10, 6))
    plt.boxplot(df[column].dropna(), vert=True, patch_artist=True)
    plt.title(f'Box Plot: {column}')
    plt.ylabel(column)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
'''
df = df.drop(['Количество стадий ГРП, шт','Азимут распространения трещины, градусы','Масса проппанта, тонн','Полудлина трещины (длина одного крыла), м','Высота трещины, м','Ширина трещины, мм','Дебит жидкости скважины, м3/сут','Дебит воды скважины, м3/сут','Депрессия (разница пластового и забойного давлений), атм'], axis = 1)
'''
df_numeric = df.select_dtypes(include=['number'])

# Матрица корреляций Пирсона
corr_matrix = df_numeric.corr()
    
# Визуализация матрицы корреляций
fig, ax = plt.subplots(figsize=(20, 15))
    
# Создаем heatmap
im = ax.imshow(corr_matrix.values, cmap='coolwarm', vmin=-1, vmax=1)
    
# Добавляем подписи
ax.set_xticks(range(len(corr_matrix.columns)))
ax.set_yticks(range(len(corr_matrix.columns)))
ax.set_xticklabels(corr_matrix.columns, rotation = 90, ha = 'right')
ax.set_yticklabels(corr_matrix.columns)
    
# Добавляем значения в ячейки
for i in range(len(corr_matrix.columns)):
    for j in range(len(corr_matrix.columns)):
        color = "white" if abs(corr_matrix.iloc[i, j]) > 0.5 else "black"
        text = ax.text(j, i, f'{corr_matrix.iloc[i, j]:.3f}',
                      ha="center", va="center", color=color, fontsize=12, fontweight='bold')
    
# Добавляем цветовую шкалу
cbar = plt.colorbar(im, ax=ax, shrink=0.8)
cbar.set_label('Коэффициент корреляции', rotation=270, labelpad=20)
 
plt.title('Матрица корреляций Пирсона числовых переменных', fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.show()


# Список ключевых признаков (X)
features = [
    'Радиус зоны дренирования, м',
    'Количество стадий ГРП, шт',
    'Мощнось продуктивного пласта, м',
    'Пластовое давление, атм',
    'Проницаемость, мД',
    'Вязкость нефти, сП',
    'Азимут распространения трещины, градусы',
    'Масса проппанта, тонн',
    'Полудлина трещины (длина одного крыла), м',
    'Высота трещины, м',
    'Ширина трещины, мм',
    'Забойное давление, атм',
    'Депрессия (разница пластового и забойного давлений), атм',
    'Дебит жидкости скважины, м3/сут',
    'Дебит воды скважины, м3/сут'
]

# Целевая переменная (y)
target = 'Дебит нефти скважины, м3/сут'

X = df[features]
y = df[target]

# --- 3. Разделение данных на обучающую и тестовую выборки ---
# 80% данных используем для обучения, 20% — для проверки качества модели
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Размер обучающей выборки: {X_train.shape[0]} записей")
print(f"Размер тестовой выборки: {X_test.shape[0]} записей")

# --- 4. Обучение модели линейной регрессии ---

model = LinearRegression()
model.fit(X_train, y_train)

print("\n--- Обучение завершено ---")

# --- 5. Оценка качества модели ---

# Прогноз на тестовых данных
y_pred = model.predict(X_test)

# Расчет метрик
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\nСреднеквадратичная ошибка (MSE) на тесте: {mse:.2f}")
print(f"Коэффициент детерминации (R^2) на тесте: {r2:.2f}")
# R^2 = 1.0 означает идеальное предсказание

# --- 6. Интерпретация весов (коэффициентов) модели ---
print("\nКоэффициенты (веса) модели для каждого признака:")
for feature, coef in zip(features, model.coef_):
    print(f"  {feature}: {coef:.4f}")

# --- 7. Пример использования модели для нового набора данных ---
new_data = {
    'Радиус зоны дренирования, м':[200],
    'Количество стадий ГРП, шт':[5],
    'Мощнось продуктивного пласта, м':[3.5],
    'Пластовое давление, атм':[330],
    'Проницаемость, мД':[1.4],
    'Вязкость нефти, сП':[0.73],
    'Азимут распространения трещины, градусы':[210.85],
    'Масса проппанта, тонн':[90],
    'Полудлина трещины (длина одного крыла), м':[158],
    'Высота трещины, м':[32],
    'Ширина трещины, мм':[0.0036],
    'Забойное давление, атм':[65],
    'Депрессия (разница пластового и забойного давлений), атм':[265],
    'Дебит жидкости скважины, м3/сут':[65],
    'Дебит воды скважины, м3/сут':[14]
}
new_df = pd.DataFrame(new_data)
prediction = model.predict(new_df)
print(f"\nПрогноз дебита нефти для нового примера: {prediction[0]:.2f} м3/сут")

plt.scatter(y_test, y_pred, alpha=0.7)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', lw=2)
plt.xlabel('Истинные значения')
plt.ylabel('Предсказанные значения')
plt.title('Сравнение истинных и предсказанных значений')
plt.show()


# Список ключевых признаков (X) теперь содержит "Депрессию"
features = [
            'Радиус зоны дренирования, м',
            'Количество стадий ГРП, шт',
            'Мощнось продуктивного пласта, м',
            'Пластовое давление, атм',
            'Проницаемость, мД',
            'Вязкость нефти, сП',
            'Азимут распространения трещины, градусы',
            'Масса проппанта, тонн',
            'Полудлина трещины (длина одного крыла), м',
            'Высота трещины, м',
            'Ширина трещины, мм',
            'Забойное давление, атм',
            'Депрессия (разница пластового и забойного давлений), атм',
            'Дебит жидкости скважины, м3/сут',
            'Дебит воды скважины, м3/сут'
]

# Целевая переменная (y)
target = 'Дебит нефти скважины, м3/сут'

X = df[features]
y = df[target]

# --- 3. Разделение данных на обучающую и тестовую выборки ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Размер обучающей выборки: {X_train.shape[0]} записей")
print(f"Размер тестовой выборки: {X_test.shape[0]} записей")

# --- 4. Обучение модели Random Forest ---

# Создаем модель Random Forest Regressor
# n_estimators=100 означает, что модель построит 100 деревьев решений
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

print("\n--- Обучение Random Forest завершено ---")

# --- 5. Оценка качества модели ---

# Прогноз на тестовых данных
y_pred = model.predict(X_test)

# Расчет метрик
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\nСреднеквадратичная ошибка (MSE) на тесте: {mse:.2f}")
print(f"Коэффициент детерминации (R^2) на тесте: {r2:.2f}")

# --- 6. Интерпретация важности признаков ---
# Random Forest дает оценку важности каждого признака
print("\nВажность признаков в модели Random Forest:")
for feature, importance in zip(features, model.feature_importances_):
    print(f"  {feature}: {importance:.4f}")

# --- 7. Пример использования модели для нового набора данных ---

new_data = {
   'Радиус зоны дренирования, м':[200],
   'Количество стадий ГРП, шт':[5],
   'Мощнось продуктивного пласта, м':[3.5],
   'Пластовое давление, атм':[330],
   'Проницаемость, мД':[1.4],
   'Вязкость нефти, сП':[0.73],
   'Азимут распространения трещины, градусы':[210.85],
   'Масса проппанта, тонн':[90],
   'Полудлина трещины (длина одного крыла), м':[158],
   'Высота трещины, м':[32],
   'Ширина трещины, мм':[0.0036],
   'Забойное давление, атм':[65],
   'Депрессия (разница пластового и забойного давлений), атм':[265],
   'Дебит жидкости скважины, м3/сут':[65],
   'Дебит воды скважины, м3/сут':[14]
}
new_df = pd.DataFrame(new_data)
prediction = model.predict(new_df)
print(f"\nПрогноз дебита нефти для нового примера: {prediction[0]:.2f} м3/сут")

plt.scatter(y_test, y_pred, alpha=0.7)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', lw=2)
plt.xlabel('Истинные значения')
plt.ylabel('Предсказанные значения')
plt.title('Сравнение истинных и предсказанных значений (Random Forest)')
plt.show()

'''

pca = PCA(n_components=2)

df_pca = pca.fit_transform(df)

explained_variance_ratio = pca.explained_variance_ratio_

print("Преобразованные данные:\n", df_pca[:5])
print("Компоненты:\n", pca.components_)
print("Доля объяснённой дисперсии:\n", explained_variance_ratio)

plt.plot(range(1, len(explained_variance_ratio) + 1), np.cumsum(explained_variance_ratio), marker='o')
plt.xlabel('Количество компонент')
plt.ylabel('Кумулятивная доля объяснённой дисперсии')
plt.title('График объяснённой дисперсии для разных компонент')
plt.grid(True)
plt.show()





# определение целевой переменной (целевой столбец)
y = df['Дебит нефти скважины, м3/сут']

# Шаг 1: Применение PCA для уменьшения размерности
pca = PCA(n_components=2)  # Оставляем 2 компоненты
df_pca = pca.fit_transform(df)

# Шаг 2: Разделяем данные на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(df_pca, y, test_size=0.2, random_state=42)

# Шаг 3: Обучение модели линейной регрессии
model = LinearRegression()
model.fit(X_train, y_train)

# Шаг 4: Оценка модели на тестовых данных
y_pred = model.predict(X_test)

# Вычисляем ошибку модели (MSE)
mse = mean_squared_error(y_test, y_pred)
print(f"Среднеквадратичная ошибка (MSE): {mse}")
r2 = r2_score(y_test, y_pred)
print(f'r2score:{r2}')
# Визуализация предсказанных и истинных значений
plt.scatter(y_test, y_pred, alpha=0.7)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', lw=2)
plt.xlabel('Истинные значения')
plt.ylabel('Предсказанные значения')
plt.title('Сравнение истинных и предсказанных значений')
plt.show()


# Шаг 3: Обучение модели случайного леса
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Шаг 4: Оценка модели на тестовых данных
y_pred = rf_model.predict(X_test)

# Вычисляем ошибку модели (MSE)
mse_rf = mean_squared_error(y_test, y_pred)
print(f"Среднеквадратичная ошибка (MSE): {mse_rf}")
r2_rf = r2_score(y_test, y_pred)
print(f'r2score:{r2_rf}')
# Визуализация предсказанных и истинных значений
plt.scatter(y_test, y_pred, alpha=0.7)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', lw=2)
plt.xlabel('Истинные значения')
plt.ylabel('Предсказанные значения')
plt.title('Сравнение истинных и предсказанных значений (Random Forest)')
plt.show()


'''


# --- 1. Данные ---

features = [
    'Радиус зоны дренирования, м',
    'Количество стадий ГРП, шт',
    'Мощнось продуктивного пласта, м',
    'Пластовое давление, атм',
    'Проницаемость, мД',
    'Вязкость нефти, сП',
    'Азимут распространения трещины, градусы',
    'Масса проппанта, тонн',
    'Полудлина трещины (длина одного крыла), м',
    'Высота трещины, м',
    'Ширина трещины, мм',
    'Забойное давление, атм',
    'Депрессия (разница пластового и забойного давлений), атм',
    'Дебит жидкости скважины, м3/сут',
    'Дебит воды скважины, м3/сут'
]

target = 'Дебит нефти скважины, м3/сут'

X = df[features]
y = df[target]

# --- 2. Разделение данных ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Размер обучающей выборки: {X_train.shape[0]} записей")
print(f"Размер тестовой выборки: {X_test.shape[0]} записей")

# --- 3. Обучение модели градиентного бустинга ---
gbr = GradientBoostingRegressor(
    n_estimators=300,       # количество деревьев
    learning_rate=0.05,      # скорость обучения
    max_depth=4,            # максимальная глубина дерева
    random_state=42
)
gbr.fit(X_train, y_train)

print("\n--- Обучение завершено ---")

# --- 4. Оценка качества модели ---
y_pred = gbr.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Среднеквадратичная ошибка (MSE) на тесте: {mse:.2f}")
print(f"Коэффициент детерминации (R^2) на тесте: {r2:.2f}")

# --- 5. Важность признаков ---
print("\nВажность признаков в модели Gradient Boosting:")
for feature, importance in zip(features, gbr.feature_importances_):
    print(f"  {feature}: {importance:.4f}")

# --- 6. Прогноз для нового примера ---
new_data = {
    'Радиус зоны дренирования, м':[200],
    'Количество стадий ГРП, шт':[5],
    'Мощнось продуктивного пласта, м':[3.5],
    'Пластовое давление, атм':[330],
    'Проницаемость, мД':[1.4],
    'Вязкость нефти, сП':[0.73],
    'Азимут распространения трещины, градусы':[210.85],
    'Масса проппанта, тонн':[90],
    'Полудлина трещины (длина одного крыла), м':[158],
    'Высота трещины, м':[32],
    'Ширина трещины, мм':[0.0036],
    'Забойное давление, атм':[65],
    'Депрессия (разница пластового и забойного давлений), атм':[265],
    'Дебит жидкости скважины, м3/сут':[65],
    'Дебит воды скважины, м3/сут':[14]
}
new_df = pd.DataFrame(new_data)
prediction = gbr.predict(new_df)
print(f"\nПрогноз дебита нефти для нового примера: {prediction[0]:.2f} м3/сут")

# --- 7. Визуализация истинных и предсказанных значений ---
plt.scatter(y_test, y_pred, alpha=0.7)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', lw=2)
plt.xlabel('Истинные значения')
plt.ylabel('Предсказанные значения')
plt.title('Сравнение истинных и предсказанных значений (Gradient Boosting)')
plt.show()


# --- 1. Данные ---
features = [
    'Радиус зоны дренирования, м',
    'Количество стадий ГРП, шт',
    'Мощнось продуктивного пласта, м',
    'Пластовое давление, атм',
    'Проницаемость, мД',
    'Вязкость нефти, сП',
    'Азимут распространения трещины, градусы',
    'Масса проппанта, тонн',
    'Полудлина трещины (длина одного крыла), м',
    'Высота трещины, м',
    'Ширина трещины, мм',
    'Забойное давление, атм',
    'Депрессия (разница пластового и забойного давлений), атм',
    'Дебит жидкости скважины, м3/сут',
    'Дебит воды скважины, м3/сут'
]

target = 'Дебит нефти скважины, м3/сут'

X = df[features]
y = df[target]

# --- 2. Разделение данных ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Размер обучающей выборки: {X_train.shape[0]} записей")
print(f"Размер тестовой выборки: {X_test.shape[0]} записей")

# --- 3. Обучение модели XGBoost ---
xgb = XGBRegressor(
    n_estimators=300,          # количество деревьев
    learning_rate=0.05,        # аналог learning_rate
    max_depth=4,               # максимальная глубина дерева
    subsample=0.9,             # доля данных для каждого дерева (регулирует overfitting)
    colsample_bytree=0.9,      # доля признаков для каждого дерева
    objective='reg:squarederror',
    random_state=42
)

xgb.fit(X_train, y_train)

print("\n--- Обучение XGBoost завершено ---")

# --- 4. Оценка качества ---
y_pred = xgb.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Среднеквадратичная ошибка (MSE) на тесте: {mse:.2f}")
print(f"Коэффициент детерминации (R^2): {r2:.2f}")

# --- 5. Важность признаков ---
print("\nВажность признаков (XGBoost):")
feature_importances = xgb.feature_importances_
for feature, importance in zip(features, feature_importances):
    print(f"  {feature}: {importance:.4f}")

# --- 6. Прогноз для нового примера ---
new_data = {
    'Радиус зоны дренирования, м':[200],
    'Количество стадий ГРП, шт':[5],
    'Мощнось продуктивного пласта, м':[3.5],
    'Пластовое давление, атм':[330],
    'Проницаемость, мД':[1.4],
    'Вязкость нефти, сП':[0.73],
    'Азимут распространения трещины, градусы':[210.85],
    'Масса проппанта, тонн':[90],
    'Полудлина трещины (длина одного крыла), м':[158],
    'Высота трещины, м':[32],
    'Ширина трещины, мм':[0.0036],
    'Забойное давление, атм':[65],
    'Депрессия (разница пластового и забойного давлений), атм':[265],
    'Дебит жидкости скважины, м3/сут':[65],
    'Дебит воды скважины, м3/сут':[14]
}

new_df = pd.DataFrame(new_data)

prediction = xgb.predict(new_df)
print(f"\nПрогноз дебита нефти для нового примера: {prediction[0]:.2f} м3/сут")

# --- 7. Визуализация ---
plt.scatter(y_test, y_pred, alpha=0.7)
plt.plot(
    [min(y_test), max(y_test)],
    [min(y_test), max(y_test)],
    color='red', lw=2
)
plt.xlabel('Истинные значения')
plt.ylabel('Предсказанные значения')
plt.title('Сравнение истинных и предсказанных значений (XGBoost)')
plt.show()





































