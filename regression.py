import numpy as np
from pandas import DataFrame, read_csv
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn import linear_model
from sklearn.preprocessing import OneHotEncoder

np.set_printoptions(precision=2)

df_cities = pd.read_csv(r'cities.csv', sep=',')
df_vacancy = pd.read_csv(r'hh_vacancy.csv', sep=',')
df_company = pd.read_csv(r'hh_company.csv', sep=',')

print(df_vacancy)
print(df_vacancy.shape)
print(df_vacancy.info())

# Разделение на обучающую и тестовую выборку по id, который соответствует порядку добавления вакаансий на сайт.
count_df = len(df_vacancy)
df_vacancy.sort_values(by='id', ascending=True, inplace=True)
df_vacancy_test = df_vacancy[int(2*count_df/3):]
df_vacancy_train = df_vacancy[:int(2*count_df/3)]

# Первичный анализ данных и формирование признаков рекомендуется проводить на обучающей выборке.
print(df_vacancy_train.describe())

print(df_company.head())

# Удаление записей с пропущенными значениями NaN.
df_vacancy_train = df_vacancy_train[np.isfinite(df_vacancy_train['salary_from']) &
                                    np.isfinite(df_vacancy_train['area_id']) &
                                    np.isfinite(df_vacancy_train['employer_id']) &
                                    np.isfinite(df_vacancy_train['specialization_id'])]

# Выбор величин x и y
train_y = df_vacancy_train['salary_from']
train_X = df_vacancy_train[['area_id', 'employer_id', 'specialization_id']]

# Обучение модели. Линейная регрессия
reg = linear_model.LinearRegression()
print(reg.fit(train_X, train_y))
print (reg.coef_) #[-2.45e+00 -3.35e-03 -1.54e+03]

# Точность на обучающей выборке.
train_r2 = r2_score(train_y, reg.predict(train_X))
print(train_r2) #0.06585492603413479

# Подготовка тестовой выборки.
df_vacancy_test = df_vacancy_test[np.isfinite(df_vacancy_test['salary_from']) &
                                  np.isfinite(df_vacancy_test['area_id']) &
                                  np.isfinite(df_vacancy_test['employer_id']) &
                                  np.isfinite(df_vacancy_test['specialization_id'])]
test_y = df_vacancy_test['salary_from']
test_X = df_vacancy_test[['area_id', 'employer_id', 'specialization_id']]

# Точность на тестовой выборке.
test_r2 = r2_score(test_y, reg.predict(test_X))
print(test_r2) #0.021286993703545587

def adjusted_r2_score(test_y, test_X, predict_y):
    '''Коэффициент детерминации, скорректированный на число признаков.'''
    n=test_X.shape[0]       # количество наблюдений
    p=test_X.shape[1] - 1   # количество признаков, включенных в модель
    r2 = r2_score(test_y, predict_y)
    adj_r2 = 1 - (1 - r2) * ((n - 1)/(n-p-1))
    return adj_r2

# Скорректированная точность на тестовой выборке.
test_adj_r2 = adjusted_r2_score(test_y, test_X, reg.predict(test_X))
print(test_adj_r2) #0.02064289169216249 плохое качество модели

# можно попробовать убрать лишние колонки и заменить строки в некоторых колонках интовыми значениями
# пробовать линейную, квадратичную, кубическую регрессии

# One Hot Encoding 
# подразумевает создание 10 признаков, все из которых равны нулю за исключением одного.
#  На позицию, соответствующую численному значению признака мы помещаем 1.
onehot_encoder = OneHotEncoder(sparse=False)

encoded_categorical_columns = pd.DataFrame(onehot_encoder.fit_transform(df_vacancy_train))
print(encoded_categorical_columns.head())
