from keras.datasets import cifar10
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras import Model

# PyTorch like network model in Keras
class MyModel(Model):
    def __init__(self):
        super(MyModel, self).__init__()
        num_classes = 10
        self.flatten = Flatten(input_shape=[28,28])
        self.fc1 = Dense(units=128, activation='relu')
        self.fc2 = Dense(units=num_classes, activation='softmax')

    def call(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        return self.fc2(x)

# Simple network model in Keras function
def MyModel2():
    net_input = tf.keras.layers.Input(shape=[28,28])
    output = tf.keras.layers.Flatten(net_input)
    output = tf.keras.layers.Dense(units=128, activation='relu')(output)
    output = tf.keras.layers.Dense(units=num_classes, activation='softmax')(output)

    model = tf.keras.models(inputs=net_input, outputs = output)
    return model

def MyModel3():
    num_classes = 10
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=[32,32,3]),
        tf.keras.layers.Dense(units=128, activation='relu'),
        tf.keras.layers.Dense(units=num_classes, activation='softmax')])
    return model


# 1. load data
(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
train_images, test_images = train_images/255.0, test_images/255.0

# 2. build model (1) use sequential model; (2) use subclass model
num_classes = 100
#model = tf.keras.models.Sequential([
#    tf.keras.layers.Flatten(input_shape=[28,28]),
#    tf.keras.layers.Dense(units=128, activation='relu'),
#    tf.keras.layers.Dense(units=num_classes, activation='softmax')])
model = MyModel3() ## MyModel2 not working
model.summary()
#model = MyModel()

# 3. train model with data
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=10)

# 4. validate model with data
model.evaluate(test_images, test_labels)        # evaluate = test

# 5. infer/predict use trained model
pred_labels = model.predict(test_images)
pred_labels_Y = np.argmax(pred_labels, axis=1)
#print(pred_labels)

#print(test_labels)
#print(pred_labels_Y)
# confusion matrix
#print(pd.crosstab(test_labels, pred_labels_Y, rownames=['label'], colnames=['predict']))

# 6. save/restore model and check the accuracy, only work in sequntial or functional models, not subclass model 
#model.save('mnist_model.h5')
#new_model = tf.keras.models.load_model('mnist_model.h5')
#new_pred_labels = new_model.predict(test_images)
#np.testing.assert_allclose(pred_labels, new_pred_labels, atol=1e-6)

# Export the model to a SavedModel for multiple platforms, only for TF2.x
#tf.keras.experimental.export_saved_model(model, 'mnist_model.h6')
#new_model = tf.keras.experimental.load_from_saved_model('mnist_model.h6')
#new_pred_labels = new_model.predict(test_images)
#np.testing.assert_allclose(pred_labels, new_pred_labels, atol=1e-6)


