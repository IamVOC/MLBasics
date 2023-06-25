# Нейронные сети
Нейронные сети довольно просты, пока дело не доходит до архитектуры. Сейчас мы дадим краткий экскурс в нейронки, а дальше лишь дело экспериментов.
## Общая информация
В отличии от предыдущих моделей здесь мы будем пользоваться не sklearn, а tensorflow (на предобработку все этой не влияет! Препроцессинги остаются такими же).
## Полносвязная нейронная сеть
Полносвязная нейронная сеть подходит и для решения задач классификации, и для решения задач регрессии. Все нейронки дальше будут написаны с использованием Sequantial API, но не забываем о существовании Subclass API и Functional API.
```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
# Здесь архитектура нейронной сети

# Компиляция и обучение
```
### Метод компиляции
```python
model.compile(args)
```
Архитектуру модели можно увидеть с помощью метода
```python
model.summary()
```
Главные аргументы:
1. optimizer
    На вход идет название или класс оптимизатора
    Самый популярный оптимизатор - adam
    Список оптимизаторов можно найти здесь: https://www.tensorflow.org/api_docs/python/tf/keras/optimizers
    (Также можете попробовать sgd, rmsprop, adagrad, adadelta, adamax, nadam)
2. loss
    На вход идет название или класс лосс функции
    Самые популярные лосс функции:
    - mean_squared_error (используется для задач регрессии)
    - binary_crossentropy (используется для задач бинарной классификации)
    - categorical_crossentropy (используется для задач классификации)
    
    Список лосс функций можно найти здесь: https://www.tensorflow.org/api_docs/python/tf/keras/losses

Также для того, чтобы не делать лишних эпох обучения можно отслеживать метрики с помощью аргумента metrics, на вход которого идет список метрик
### Метод обучения
На вход идут X и y - обучающие выборки, epochs - кол-во эпох обучения и batch_size - Размер батча, то есть кол-во объектов используемых на каждом обновлении градиента
Полную информацию об аргументах методов и модели можно найти здесь: https://www.tensorflow.org/api_docs/python/tf/keras/Model
### Слои
Новые слои добавляются методом 
```python 
model.add(слой)
```
1. Dense
    1. Размерность выходного вектора
    2. Функция активации
        Самые популярные: tanh, relu, sigmoid

    Для входного слоя также задаются input_shape=() - размерность признакового пространства X, а для выходного:
    1. Для регрессии:
        Размерность выходного слоя - размерность вектора целевых признаков (1) и без функции активации
    2. Для бинарной классификации:
        Размерность выходного слоя - 1 с функцией активации sigmoid
    3. Для классификации
        Размерность выходного слоя - кол-во классов с функцией активации softmax
### Примеры
Регрессия
```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(input_dim,)))
model.add(Dense(64, activation='relu'))
model.add(Dense(1))  # Выходной слой без функции активации для регрессии

model.compile(optimizer='adam', loss='mean_squared_error')

model.fit(x_train, y_train, batch_size=32, epochs=10, validation_data=(x_val, y_val))
```
Бинарная классификация
```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(input_dim,)))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))  # Выходной слой с функцией активации sigmoid для бинарной классификации

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=32, epochs=10, validation_data=(x_val, y_val))
```
Классификация
```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(input_dim,)))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))  # Выходной слой с функцией активации sigmoid для бинарной классификации

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=32, epochs=10, validation_data=(x_val, y_val))
```
## Сверточная неронная сеть (CNN)
Сверточная нейронная сеть используется для задач классификации с картинками
### Слои
Кроме слоев Dense также перед ними появляются 3 вида слоев:
1. Convolutional Layer
    Используются для свертки картинки, чтобы обнаружить локальные паттерны
    1. Размерность выходного вектора
    2. kernel_size
        На вход принимается вектор из двух чисел - размерность ядра по которому идет свертка
    3. strides
        На вход принимается шаг с которым будет происходить обход по матрице
    4. Функция активации
        Самые популярные: tanh, relu, sigmoid
2. Pooling Layer
    Используется для понижение размерности картинки
    1. pool_size
        На вход принимается вектор из двух чисел - размерность пула
3. Flatten 
    Преобразует картинку в вектор для последующего обучения задачи категоризации с помощью Dense
### Примеры
Простая сверточная нейросеть
```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(height, width, channels)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=32, epochs=10, validation_data=(x_val, y_val))
```
Более сложный вариант сверточной нейронной сети
```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(height, width, channels)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=32, epochs=10, validation_data=(x_val, y_val))
```
## Фишки с нейронными сетями
### Заморозка слоев
Вы можете заморозить некоторые слои, чтобы не обучать их:
```python
model.layers[0].trainable = False # Замораживает первый слой

for layer in model.layers[:x]:
    layer.trainable = False #Замораживает все слои до x-ого
```
### Использование готовых моделей
Вы можете использовать уже готовые модели. К примеру в tensorflow есть модель VGG19, которая решает задачу классификации с картинками. Чтобы заного обучить ее на своих картинках заморозьте лишние слои и обучите.
### Защита от переобучения
С помощью слоя Dropout(percentage) применяет случайное отключение нейронов, что может защитить от переобучения
