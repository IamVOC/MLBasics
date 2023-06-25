# Предобработка данных
Предобработка после загрузки данных имеет в себе множество этапов. Возьмем основные из них:
- [Изучение данных](#header1)
- [Очистка данных](#header2)
- [Кодирование категориальных данных](#header3)
- [Масштабирование данных](#header4)
- [Разбиение данных](#header5)
- [Понижение размерности](#header6)
- [Обработка текстовых данных](#header7)

<a id="header1"></a>
# Изучение данных 
1. Получим общую информацию о датасете от которой можно отталкиваться:
    ```python
    df.head() # Выводит первые строки данных для того, чтобы ознакомится со структурой датасета
    df.info() # Выводит столбцы, кол-во ненулевых данных в каждом из столбцов и тип данных столбца
    df.describe() # Выводит описательные статистики
    df.value_counts() # Подсчитывает уникальные значения
    df.isnull().sum() # Подсчитывает кол-во null значений(так же с df.isna().sum())
    ```
2. Изучим данные с помощью некоторых графиков matplotlib
    1. Гистограмма
        ```python
        plt.hist(df['chosen_col'], bins=30) # С bins нужно поиграться, потому что при низком кол-ве сложно понять форму распределения, а при большом на графике выскакивают редкие значения
        ```
        Позволяет определить форму распределения, а также выбросы и аномалии выбраного столбца
       
    3. Ящик с усами
        ```python
        plt.boxplot(df['chosen_col'])
        ```
        Представляет распределение данных через квартили, медиану и выбросы
       
    4. Линейный график
        ```python
        plt.plot(df['first_col'], df['second_col'])
        ```
        Показывает изменение значения переменной по другой независимой переменной
       
    5. Диаграмма рассеяния
        ```python
        plt.scatter(df['first_col'], df['second_col'])
        ```
        Помогает идентифицировать корреляцию, кластеры, выбросы
       
    6. Тепловые карты
        Позволяют быстро выявить паттерны, тенденции и различия в данных
        Очень полезна в связке с матрицей корреляции 
        ```python
        sns.heatmap(df.corr())
        ```

<a id="header2"></a>
# Очистка данных
1. Обработаем пропущенные значения
    ```python
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    # Стоит принять во внимание, что если NaN в данном случае характеризует отсутствие чего-либо(к примеру кол-во объектов) или в столбце переменные бинарные, то стоит заменить на 0
    df[cat_cols] = df[cat_cols].fillna(df[cat_cols].mode().iloc[0])
    ```
    Мы заменяем NaN и Null, на самые вероятные значения для числовых и категориальных признаков
   
2. Удаление дубликатов
    ```python
    df.drop_duplicates()
    ```
    Модель может очень сильно заболеть, если в датасете останутся дубликаты
   
3. Хэндлинг выбросов
    ```python
    df = df[df['col'] < 777]
    ```
    Хоть модель часто и может справится с небольшим кол-вом выбросов, но иногда стоит перестраховаться, чтобы не получить болеющую модель
   
5. Хэндлинг несоответствий
    ```python
    df.loc[df['col'] < 0, 'col'] = 0
    ```
    Стоит не забывать о возможных несоответствиях в датасете, которые появились случайно (в данном случае col должен быть неотрицательным)
    
6. Удаление бесполезных данных
    ```python
    df.drop(['col1', 'col2'], axis=1)
    ```
    Если матрица корреляций или диаграмма рассеяний показывают ужасный результат с целевым признаком, то удаляйте, ведь эти признаки повлияют лишь на то насколько долго будет обучаться модель(Если такой признак категориальный, то там не стоит задумываться, ведь чаще всего он породит множество столбцов с помощью One-of-K)
    
<a id="header3"></a>
# Кодирование категориальных данных
1. One-Hot-Encoding
    ```python
    ohe = pd.get_dummies(df[cat_cols])
    df = pd.concat([df, ohe], axis=1)
    df = df.drop(cat_cols, axis=1)
    ```
    Используется, когда переменная не имеет порядка или необходимо избежать создания ложной упорядоченности
    (Стоит быть осторожным, ведь One-of-K при высокой кардинальности может заразить датасет огромным кол-вом столбцов)
    
2. Label Encoding
    ```python
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    for c in cat_cols:
        df[c+'encoded'] = le.fit_transform(df[c])
    df = df.drop(cat_cols, axis=1)
    ```
    Используется, когда есть явный порядок(Часто спасает при высокой кардинальности признака)
    
3. Frequency Encoding
    ```python
    for c in cat_cols:
        df[c+'encoded'] = df[c].map(df[c].value_counts(normilize=True).to_dict())
    df = df.drop(cat_cols, axis=1)
    ```
    Используется, когда есть зависимость целевого признака от частоты появления категории(Часто спасает при высокой кардинальности)
    
<a id="header4"></a>
# Масштабирование данных
Все что требуется для масштабирования находится в sklearn.preprocessing и имеет следующий вид:
```python
scaler = ChosenScaler()
X_scaled = scaler.fit_transform(X)
```
Сам скейлинг чаще всего происходит на данных без целевого признака
1. Min-Max Scaling
    Сохраняет относительное распределение значений. Полезно для методов, использующих расстояние между точками

2. Standartization
    Признаки имеют нормальное распределение после масштабирования. Полезно для моделей, которые предполагают, что признаки распределены нормально

3. Normalization
    Признаки сохраняют форму распределения данных. Полезно для методов на основе длины векторов(к примеру SVM)

4. Log Transformation
    Вместо импорта скейлера используется следующий вид:
    ```python
    X_scaled = np.log(X)
    ```
    Сглаживает экстремальные значения и может приблизить данные к нормальному распределению
    
5. Robust Scaling
    Устойчиво к выбросам, поэтому используется с моделями, которые чувствительны к ним

<a id="header5"></a>
# Разбиение данных
1. Simple Random Split
    ```python
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=777) # Чаще всего используют разбиение 80/20 или 70/30. В связи с этим меняется test_size
    ```
    Разбивает выборку в нужном соотношении. Самый простой вариант

2. Stratified split
    ```python
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=df[classes] random_state=777)
    ```
    Разбивает выборку, сохраняя пропорции несбалансированных классов в обучающей и тестовой выборках
    
3. Cross-Validation
    ```python
    from sklearn.model_selection import cross_val_score
    scores = cross_val_score(model, X, y, cv=5)
    ```
    Модель обучается на одном фолде и тестируется на остальных, что позволяет точно оценить производительность модели
    
<a id="header6"></a>
# Понижение размерности данных
1. PCA
    ```python
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)# Будущая размерность
    X_pca = pca.fit_transform(X)
    ```
    Преобразует признаки в ортогональные переменные, которые упорядочены по обьясненной ими дисперсии.
    
2. t-SNE
    ```python
    from sklearn.manifold import TSNE
    tsne = TSNE(n_components=2, random_state=777)# Будущая размерность
    X_tsne = tsne.fit_transform(X)
    ```
    Сохраняет локальные связи между данными. Подходит для визуализации кластеров
    
3. LDA
    ```python
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    lda = LinearDiscriminantAnalysis(n_components=2)# Будущая размерность
    X_lda = lda.fit_transform(X)
    ```
    Ищет новые оси признаков

<a id="header7"></a>
# Обработка текстовых данных
Если задача будет связана с текстовыми данными(что на вряд-ли). Если меч пригодится один раз в жизни, носить его нужно всегда. Вот и мы будем носить эту инфу.
1. Токенизация
    ```python
    from nltk.tokenize import word_tokenize
    tokens = word_tokenize(text)
    ```
    Делит текст на токены, которые используются обработчиками и моделями

2. Удаление стоп-слов
    ```python
    from nltk.corpus import stopwords
    stop_words = set(stopwords.words('russian'))
    filtered = [word for word in tokens if word.lower() not in stop_words]
    ```
    Убирает, слова не влияющие на смысл документа
    
3. Приведение к нижнему регистру
    ```python
    ltext = text.lower()
    ```
    Приводит все слова к нижнему регистру
    
4. Лемматизация
    ```python
    from nltk.stem import WordNetLemmatizer
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    ```
    Приводит слова к базовой форме
    
5. Векторизация
    1. Bag-of-Words
        ```python
        from sklearn.feature_extraction.text import CountVectorizer
        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform(corpus)
        # Матрица счетчиков и список слов
        X.toarray()
        vectorizer.get_feature_names()
        ```
        
    2. TF-IDF
        ```python
        from sklearn.feature_extraction.text import TfidfVectorizer
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(corpus)
        # Матрица счетчиков и список слов
        X.toarray()
        vectorizer.get_feature_names()
        
