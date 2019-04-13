import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 1. Generate data
points = 1000
X = np.linspace(-10, 10, points)
np.random.shuffle(X)
Y = np.heaviside( 1.0*X + np.random.normal(0, 3, (points, )), 0)
x_train, y_train = X[int(points*0.1):], Y[int(points*0.1):]
x_test, y_test = X[:int(points*0.1)], Y[:int(points*0.1)]
#print(x_train, y_train)
#print(x_test, y_test)

# 2. Build model
model = tf.keras.models.Sequential(
    [tf.keras.layers.Dense(units=1, input_shape=[1], activation='sigmoid')])

# 3. Train model with data
model.compile(optimizer='sgd', loss='binary_crossentropy')
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
plt.title('Logistic Regression')
plt.xlabel('X')
plt.ylabel('Y')
plt.axis([-10, 10, -0.1, 1.1])
plt.show()
