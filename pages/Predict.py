import streamlit as st
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib as jl

st.set_page_config(
    page_title="Получение предсказаний моделей",
)

get_model = {
        'Линейная регрессия': 'dt.joblib',
        'Градиентный бустинг': 'gradboost.joblib',
        'Нейронная сеть': 'neural.joblib'
        }

option = st.selectbox("Выберите модель", 
                      ('Линейная регрессия', 'Градиентный бустинг', 'Нейронная сеть')
                      )
model = None

try:
    model = open(get_model[f'{option}'], 'rb')
    st.sidebar.success(f'Модель \"{option}\" успешно загружена')
except:
    st.sidebar.error(f'Модели нет или она не успела загрузиться')

upload = st.file_uploader("Выберите CSV для предсказания", type='csv')

if upload is not None and model is not None:
    csv = pd.read_csv(upload)
    y = csv['alcohol']
    x = csv.drop(['alcohol'], axis=1)
    if option == 'Нейронная сеть':
        x.reset_index(drop= True , inplace= True)
        sc = MinMaxScaler()
        x = sc.fit_transform(x) 
    jobmodel = jl.load(model)
    y_pred = jobmodel.predict(x)
    datatab = pd.DataFrame(
                data=[str(y_pred), jobmodel.score(x, y)],
                index=['Предсказание', 'Оценка'],
                columns=['']
    )
    st.table(datatab)

