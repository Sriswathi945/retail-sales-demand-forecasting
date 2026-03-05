import pandas as pd
from sklearn.linear_model import LinearRegression

data = pd.read_csv("sales_data.csv")

X = data[["month"]]
y = data["sales"]

model = LinearRegression()
model.fit(X, y)

prediction = model.predict([[13]])
print("Predicted sales:", prediction)
