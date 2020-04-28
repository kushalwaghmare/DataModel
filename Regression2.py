import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import statsmodels.api as sm

data = pd.read_csv(r'\data\Sample_Book1.csv', encoding = "utf8")

#Print Data Head
print (data.head())

#Visualize data
plt.figure(figsize=(16, 8))
plt.scatter(
    data['AGE'],
    data['UNMARRIED'],
    c='black'
)
plt.xlabel("Age")
plt.ylabel("Unmarried")
plt.show()

# Linear Approximation of the Data
IndVar = data['AGE'].values.reshape(-1,1)
DepVar = data['UNMARRIED'].values.reshape(-1,1)

reg = LinearRegression()
reg.fit(IndVar,DepVar)

# The coefficients
print('Intercept: \n', reg.intercept_[0])
print('Coefficient: \n', reg.coef_[0][0])

#Visualize the regression line
"""
predictions = reg.predict(IndVar)

plt.figure(figsize = (16,8))
plt.scatter(
    data['AGE'],
    data['UNMARRIED'],
    c = 'green'
)

plt.plot(
    data['AGE'],
    predictions,
    c = 'red'
)
plt.xlabel("Age")
plt.ylabel("Unmarried")
plt.show()
"""

x = data['AGE']
y = data['UNMARRIED']

print ("X: ")
print (x)

x2 = sm.add_constant(x)
print("X2")
print(x2)
est = sm.OLS(y, x2)
est2 = est.fit()
print(est2.summary())

print ("Done.")
