# Metrics

- Для оценки качетсва моделей классификации применяются следующие метрики:

accuracy -- (количество верно классифицированных объектов) / (общее количество объектов)
```python
from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_bin_knn_pred)
```

- Confusion matrix

![Текст с описанием картинки](прог_2.png)

```python
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, y_bin_knn_pred)
```
- Precision, Recall, F-мера
P  – число истинных результатов, P=TP+FN
N – число ложных результатов, N=TN+FP.

![Текст с описанием картинки](прог_3.png)

```python
from sklearn.metrics import classification_report
print(classification_report(y_test, y_bin_knn_pred))
```
