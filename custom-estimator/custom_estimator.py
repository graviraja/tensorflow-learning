import tensorflow as tf
import iris_data


def my_model_fn(features, labels, mode, params):
    net = tf.feature_column.input_layer(features, params['feature_columns'])
    for units in params['hidden_units']:
        net = tf.layers.dense(net, units, activation=tf.nn.relu)

    logits = tf.layers.dense(net, params['n_classes'], activation=None)
    predicted_classes = tf.argmax(logits, 1)

    # PREDICT Mode
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'class_ids': predicted_classes[:, tf.newaxis],
            'probabilities': tf.nn.softmax(logits),
            'logits': logits
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    # loss calculation
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    # accuracy calculation
    accuracy = tf.metrics.accuracy(labels=labels, predictions=predicted_classes)

    metrics = {'accuracy': accuracy}
    tf.summary.scalar('accuracy', accuracy[1])

    # EVAL Mode
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)

    # TRAIN Mode
    # optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate=0.05)
    # training op
    training_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=training_op)


def main():
    train_data, train_label, test_data, test_label = iris_data.load_data()
    my_feature_columns = []
    for key in train_data.keys():
        my_feature_columns.append(tf.feature_column.numeric_column(key))

    classifier = tf.estimator.Estimator(
        model_fn=my_model_fn,
        params={
            'feature_columns': my_feature_columns,
            'hidden_units': [10, 10],
            'n_classes': 3
        }
    )

    classifier.train(input_fn=lambda: iris_data.train_input_fn(train_data, train_label, 32), steps=100)
    eval_result = classifier.evaluate(input_fn=lambda: iris_data.eval_input_fn(test_data, test_label, 32))
    print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))

    expected = ['Setosa', 'Versicolor', 'Virginica']
    predict_x = {
        'SepalLength': [5.1, 5.9, 6.9],
        'SepalWidth': [3.3, 3.0, 3.1],
        'PetalLength': [1.7, 4.2, 5.4],
        'PetalWidth': [0.5, 1.5, 2.1],
    }

    predictions = classifier.predict(
        input_fn=lambda: iris_data.eval_input_fn(predict_x, labels=None, batch_size=32))

    for pred_dict, expec in zip(predictions, expected):
        template = ('\nPrediction is "{}" ({:.1f}%), expected "{}"')

        class_id = pred_dict['class_ids'][0]
        probability = pred_dict['probabilities'][class_id]

        print(template.format(iris_data.SPECIES[class_id],
                              100 * probability, expec))

if __name__ == '__main__':
    main()