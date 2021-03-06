import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def lin_reg_load_data():
    points = 500
    X = np.linspace(-10, 10, points)
    np.random.shuffle(X)
    Y = 2.0*X + 3.0 + np.random.normal(0, 1, (points, ))
    x_train, y_train = X[int(points*0.1):], Y[int(points*0.1):]
    x_test, y_test = X[:int(points*0.1)], Y[:int(points*0.1)]
    return (x_train, y_train), (x_test, y_test)


# 1. Generate data
(x_train, y_train), (x_test, y_test) = lin_reg_load_data()
#print(x_train, y_train)
#print(x_test, y_test)

# 2. Build model
model = tf.keras.models.Sequential(
    [tf.keras.layers.Dense(units=1, input_shape=[1])])
model.summary()

# 3. Train model with data
model.compile(optimizer='sgd', loss='mse')
model.fit(x_train, y_train, epochs=100)

# 4. Test model with data
model.evaluate(x_test, y_test)

# 5. Predict/Infer data
x_test_sort = np.sort(x_test)
y_pred = model.predict(x_test_sort)
#print(y_pred)

plt.scatter(x_test, y_test, c='r')
#plt.scatter(x_test_sort, y_pred, c='g')
plt.plot(x_test_sort, y_pred)
plt.grid()
plt.title('Linear Regression')
plt.xlabel('X')
plt.ylabel('Y')
plt.axis([-10, 10, -20, 25])
plt.show()
