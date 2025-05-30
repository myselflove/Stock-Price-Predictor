import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Simulated stock data
data = pd.DataFrame({
    'day': np.arange(1, 11),
    'price': [100, 102, 101, 105, 107, 110, 115, 117, 120, 125]
})

X = data[['day']]
y = data['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Evaluation
print("RMSE:", mean_squared_error(y_test, y_pred, squared=False))

# Visualization
plt.scatter(X, y, color='blue')
plt.plot(X, model.predict(X), color='red')
plt.title("Stock Price Prediction")
plt.xlabel("Day")
plt.ylabel("Price")
plt.savefig("price_prediction_chart.png")
plt.show()
