import tensorflow as tf

mnist = tf.keras.datasets.mnist

# load data
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images, test_images = train_images/255.0, test_images/255.0

# build model
num_classes = 10
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=[28,28]),
    tf.keras.layers.Dense(units=128, activation='relu'),
    tf.keras.layers.Dense(units=num_classes, activation='softmax')])

# train model with data
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=3)

# test model with data
model.evaluate(test_images, test_labels)        # evaluate = test

pred_labels = model.predict(test_images)  # predict is for 1 test
print(pred_labels)
