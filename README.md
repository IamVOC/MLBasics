# Metrics

Для оценки качетсва моделей регрессии применяются следующие метрики:
- средняя абсолютная ошибка (Mean Absolute Error, MAE);
- средняя квадратичная ошибка (Mean Squared Error, MSE);
- квадратный корень из средней квадратичной ошибки (Root Mean Squared Error);
- средняя абсолютная ошибка в процентах (Mean Absolute Percentage Error, MAPE);
- коэффициент детерминации ($R^2$).

![Текст с описанием картинки](5pYgG9-Xk_U.jpg)

```python
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from math import sqrt
print(f'MAE: {mean_absolute_error(y_test, y_pred)}')
print(f'MSE: {mean_squared_error(y_test, y_pred)}')
print(f'RMSE: {sqrt(mean_squared_error(y_test, y_pred))}')
print(f'MAPE: {sqrt(mean_absolute_percentage_error(y_test, y_pred))}')
print(f'R^2: {lr.score(X_test, y_test)}')
```
