# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries.
2. Load the house dataset with features (house size and number of rooms) and target values (house price and number of occupants), then splits the data into training and testing sets.
3. Scales the input features using StandardScaler and create an SGDRegressor with specified parameters.
4. Train the model and calculate the Mean Squared Error.
5. Display the predicted and actual values.

## Program:
```
/*
Program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor.
Developed by: Raaghavi S
RegisterNumber: 25012715  
*/

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

X = np.array([
    [800, 2],
    [1000, 3],
    [1200, 3],
    [1500, 4],
    [1800, 4],
    [2000, 5],
    [2200, 5],
    [2500, 6]
])

y = np.array([
    [40, 2],
    [55, 3],
    [65, 3],
    [85, 4],
    [95, 4],
    [110, 5],
    [125, 5],
    [145, 6]
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

sgd = SGDRegressor(max_iter=2000, eta0=0.01, learning_rate='constant', random_state=42)
model = MultiOutputRegressor(sgd)
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)

print("Predicted [Price, Occupants]:")
print(y_pred)

print("\nActual [Price, Occupants]:")
print(y_test)

mse = mean_squared_error(y_test, y_pred)
print("\nMean Squared Error:",mse)

new_house = np.array([[1600, 4]])
new_house_scaled = scaler.transform(new_house)
new_prediction = model.predict(new_house_scaled)

print("\nFor New House [1600 sq ft, 4 rooms]:")
print("Predicted House Price (lakhs):", round(new_prediction[0][0], 2))
print("Predicted Number of Occupants:", round(new_prediction[0][1]))

plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.scatter(X_test[:, 0], y_test[:, 0], color='green', label="Actual Price")
plt.scatter(X_test[:, 0], y_pred[:, 0], color='red', marker='x', label="Predicted Price")
plt.xlabel("House Size (sq ft)")
plt.ylabel("House Price (lakhs)")
plt.title("House Size vs House Price(Actual vs Predicted)")
plt.legend()
plt.grid(True)


plt.tight_layout()
plt.show()

```

## Output:
<img width="1042" height="789" alt="Screenshot 2026-01-31 115414" src="https://github.com/user-attachments/assets/5641942f-a2f3-4dc2-b862-a961d7a5a89a" />

<img width="995" height="715" alt="Screenshot 2026-01-31 115441" src="https://github.com/user-attachments/assets/6eda1f54-7336-41a4-83ba-538c26b4d6ab" />


## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
