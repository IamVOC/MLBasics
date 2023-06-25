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
- ROC-AUC score  
Полное название ROC — Receiver Operating Characteristic (рабочая характеристика приёмника). Впервые она была создана для использования радиолокационного обнаружения сигналов во время Второй мировой войны. США использовали ROC для повышения точности обнаружения японских самолетов с помощью радара. Поэтому ее называют рабочей характеристикой приемника.
AUC или area under curve — это просто площадь под кривой ROC.
True Positive Rate (TPR) показывает, какой процент среди всех positive верно предсказан моделью. TPR=TP/(TP+FN)
False Positive Rate (FPR): какой процент среди всех negative неверно предсказан моделью. FPR=FP/(TP+FN)

```python
import matplotlib.pyplot as plt
from sklearn.metrics import plot_roc_curve
plot_roc_curve(knn, X_test, y_test)
```
![Текст с описанием картинки](download.png)
