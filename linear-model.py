import tensorflow as tf
import numpy as np
from tqdm import tqdm

# importing the data
from tensorflow.examples.tutorials.mnist import input_data
data = input_data.read_data_sets("data/MNIST/", one_hot=True)

# declaring the sizes for data
img_size = 28
img_size_falt = img_size * img_size
img_shape = (img_size, img_size)

num_classes = 10


def get_input_layer():
    x = tf.placeholder(dtype=tf.float32, shape=[None, img_size_falt], name="input_image")
    y_true = tf.placeholder(dtype=tf.float32, shape=[None, num_classes], name="input_label")
    y_true_class = tf.placeholder(dtype=tf.int64, shape=[None], name="input_label_class")
    return (x, y_true, y_true_class)


def get_model_parameters():
    weights = tf.Variable(tf.zeros(shape=[img_size_falt, num_classes]), name="weights")
    bias = tf.Variable(tf.zeros(shape=[num_classes]), name="bias")
    return (weights, bias)


def linear_model(inputs, model_params, epochs, batch_size):
    x, y_true, y_true_class = inputs
    weights, bias = model_params

    # calculating the logits
    logits = tf.matmul(x, weights) + bias

    # calculating the predictions
    y_pred = tf.nn.softmax(logits)
    y_pred_class = tf.argmax(y_pred, 1)

    # defining the loss
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_true, name="loss")
    total_loss = tf.reduce_mean(loss)

    # defining the optimizer
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(total_loss)

    # calculating the accuracy
    correct_predictions = tf.equal(y_pred_class, y_true_class)
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

    # calculating number of iterations for each epoch
    num_iterations = len(data.train.labels) // batch_size
  
    # creating a session
    sess = tf.Session(graph=tf.get_default_graph())
    sess.run(tf.global_variables_initializer())

    # running the model
    for i in range(epochs):
        for step in tqdm(range(num_iterations)):
            x_batch, y_label_batch = data.train.next_batch(batch_size)
            sess.run(optimizer, feed_dict={x: x_batch, y_true: y_label_batch})
    
    # test accuracy
    data.test.cls = np.array([label.argmax() for label in data.test.labels])
    test_accuracy = sess.run(accuracy, feed_dict={x: data.test.images, y_true: data.test.labels, y_true_class: data.test.cls})
    print("Accuracy on test set : %f", (test_accuracy))


if __name__ == '__main__':
    epochs = 100
    batch_size = 100
    model_inputs = get_input_layer()
    model_params = get_model_parameters()
    linear_model(model_inputs, model_params, epochs, batch_size)