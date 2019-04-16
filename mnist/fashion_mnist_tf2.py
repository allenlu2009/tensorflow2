import tensorflow as tf
import numpy as np
import pandas as pd

fashion_mnist = tf.keras.datasets.fashion_mnist

# 1. load data
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
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
pred_labels = model.predict(test_images)
pred_labels_Y = np.argmax(pred_labels, axis=1)
#print(pred_labels)

#print(test_labels)
#print(pred_labels_Y)
# confusion matrix
print(pd.crosstab(test_labels, pred_labels_Y, rownames=['label'], colnames=['predict']))

# 6. save/restore model and check the accuracy 
model.save('fashion_mnist_model.h5')
new_model = tf.keras.models.load_model('fashion_mnist_model.h5')
new_pred_labels = new_model.predict(test_images)
np.testing.assert_allclose(pred_labels, new_pred_labels, atol=1e-6)

# Export the model to a SavedModel for multiple platforms, only for TF2.x
#tf.keras.experimental.export_saved_model(model, 'fashion_mnist_model.h6')
#new_model = tf.keras.experimental.load_from_saved_model('fashion_mnist_model.h6')
#new_pred_labels = new_model.predict(test_images)
#np.testing.assert_allclose(pred_labels, new_pred_labels, atol=1e-6)


