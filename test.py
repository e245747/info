import regression
import datasets

X, Y = datasets.load_linear_example1()

model = regression.LinearRegression()
model.fit(X, Y)

# ver.3（予測）
print(model.predict(X))

# ver.4（誤差）
print(model.score(X, Y))