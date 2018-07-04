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
    pass
