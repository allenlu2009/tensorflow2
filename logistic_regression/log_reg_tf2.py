### TF2.0 Logistic Regressionx_train_ = tf.convert_to_tensor(train_inputs_of_numpy)
y_train_ = tf.convert_to_tensor(train_labels_of_numpy)

# Logistic regression model
model = tf.keras.layers.Dense(output_size, activation="sigmoid")

# loss function
def loss_fn(model, x, y):
    predict_y = model(x)

    # return shape is (x.shape[0], ) ; each element is each data's loss.
    return tf.keras.losses.binary_crossentropy(y, predict_y)

def accuracy_fn(model, x, y):
    predict_y = model(x)

    # return shape is (x.shape[0], ) ; each element is 1 or 0.(If y[i] == y_pre[i], the i th element is 1). 
    return tf.keras.metrics.binary_accuracy(y, predict_y)

# optimizer
optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)


for epoch in range(num_epochs):
    with tf.GradientTape() as tape:
        loss = loss_fn(model, x_train_, y_train_)
    grads = tape.gradient(loss, model.variables)
    
    accuracy = accuracy_fn(model=model, x=x_train_, y=y_train_)
    
    if (epoch+1) % 5 == 0:
        print(
            "loss: {:0.3f},  acc: {}".format(
                tf.reduce_sum(loss).numpy(),   # using TF function or numpy method
                accuracy.numpy().mean()         # both of ok.
            )  
        ) 
    # update prameters using grads
    optimizer.apply_gradients(zip(grads, model.variables))
