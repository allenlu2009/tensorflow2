import tensorflow as tf
import numpy as np

mnist = tf.keras.datasets.mnist

# 1. load data
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images, test_images = train_images/255.0, test_images/255.0

# 2. build model
num_classes = 10
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=[28,28]),
    tf.keras.layers.Dense(units=128, activation='relu'),
    tf.keras.layers.Dense(units=num_classes, activation='softmax')])
model.summary()

# 3. train model with data
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=3)

# 4. validate model with data
model.evaluate(test_images, test_labels)        # evaluate = test

# 5. infer/predict use trained model
pred_labels = model.predict(test_images)  # predict is for 1 test
#print(pred_labels)

# 6. save/restore model and check the accuracy 
model.save('mnist_model.h5')
new_model = tf.keras.models.load_model('mnist_model.h5')
new_pred_labels = new_model.predict(test_images)
np.testing.assert_allclose(pred_labels, new_pred_labels, atol=1e-6)

