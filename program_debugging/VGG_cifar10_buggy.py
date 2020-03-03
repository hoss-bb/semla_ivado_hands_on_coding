import tensorflow as tf
from tensorflow.keras.datasets import cifar10
import click
import numpy as np
from tensorflow.python.training.basic_session_run_hooks import _as_graph_element
from PIL import Image

class DataLoader:
    
    def __init__(self, raw_data, batch_size, augment=False, is_training=False):
        X, y = raw_data
        y = y.squeeze()
        dset = tf.data.Dataset.from_tensor_slices((X, y))
        dset = dset.map(self.preprocess_data)
        if augment:
            dset = dset.map(self.augment_data)
        dset = dset.shuffle(10000)
        dset = dset.batch(batch_size)
        self.dset = dset 
        self.iterator = self.dset.make_initializable_iterator()
        self.next_batch = self.iterator.get_next()

    def init(self, sess):
        sess.run(self.iterator.initializer)

    def preprocess_data(self, *data_point):
        image, label = data_point
        # Convert to float
        prep_image = tf.image.convert_image_dtype(tf.reshape(image, [32, 32, 3]), tf.float32)
        # Scale to [0, 1) 
        prep_image = prep_image / 255.
        onehot_label = tf.cast(tf.one_hot(label, 10), tf.float32)
        return prep_image, onehot_label

    def augment_data(self, *data_point):
        image, label = data_point
        image_shape = tf.shape(image)
        height, width = image_shape[0], image_shape[1]
        width = tf.cast(width, tf.float32)
        height = tf.cast(height, tf.float32)

        # rotate randomly the image
        angle_rad = 1.57
        angles = tf.random_uniform([], -angle_rad, angle_rad)
        f = tf.contrib.image.angles_to_projective_transforms(angles, height, width)
        new_image = tf.contrib.image.transform(image, f)

        # crop randomly the image
        crop = 0.1
        crop_value = tf.random_uniform([], crop, 1.0)
        crop_size = tf.floor(32 * crop_value)
        cropped = tf.random_crop(new_image, [crop_size, crop_size, 3])
        new_image = tf.image.resize_images(tf.expand_dims(cropped, 0), [32, 32])[0]

        return new_image, label

def build_model(features, scope='VGG', reuse=False, training=True):
    with tf.variable_scope(scope, reuse=reuse):
        conv1 = tf.layers.conv2d(inputs=features,
                                filters=32,
                                kernel_size=[3, 3], 
                                padding='same',
                                activation=tf.nn.relu,
                                use_bias=False,
                                kernel_initializer=tf.initializers.he_normal())
        conv1_norm = tf.layers.batch_normalization(conv1, training=training)
        conv2 = tf.layers.conv2d(inputs=conv1_norm,
                                    filters=32,
                                    kernel_size=[3, 3], 
                                    padding='same',
                                    activation=tf.nn.relu,
                                    use_bias=False,
                                    kernel_initializer=tf.initializers.he_normal())
        conv2_norm = tf.layers.batch_normalization(conv2, training=training)
        pool2 = tf.layers.max_pooling2d(inputs=conv2_norm, pool_size=[2, 2], strides=2)
        reg_pool2 = tf.layers.dropout(inputs=pool2, rate=0.2, training=training)
        conv3 = tf.layers.conv2d(inputs=reg_pool2, 
                                    filters=64, 
                                    kernel_size=[3, 3],
                                    padding='same', 
                                    activation=tf.nn.relu,
                                    use_bias=False,
                                    kernel_initializer=tf.initializers.he_normal())
        conv3_norm = tf.layers.batch_normalization(conv3, training=training)
        conv4 = tf.layers.conv2d(inputs=conv3_norm, 
                                    filters=64, 
                                    kernel_size=[3, 3],
                                    padding='same', 
                                    activation=tf.nn.relu,
                                    use_bias=False,
                                    kernel_initializer=tf.initializers.he_normal())
        conv4_norm = tf.layers.batch_normalization(conv4, training=training)
        pool4 = tf.layers.max_pooling2d(inputs=conv4_norm, pool_size=[2, 2], strides=2)
        reg_pool4 = tf.layers.dropout(inputs=pool4, rate=0.3, training=training)
        conv5 = tf.layers.conv2d(inputs=features, 
                                    filters=128, 
                                    kernel_size=[3, 3],
                                    padding='same', 
                                    activation=tf.nn.relu,
                                    use_bias=False,
                                    kernel_initializer=tf.initializers.he_normal())
        conv5_norm = tf.layers.batch_normalization(conv5, training=training)
        conv6 = tf.layers.conv2d(inputs=conv5_norm, 
                                    filters=128, 
                                    kernel_size=[3, 3],
                                    padding='same', 
                                    activation=tf.nn.relu,
                                    use_bias=False,
                                    kernel_initializer=tf.initializers.he_normal())
        conv6_norm = tf.layers.batch_normalization(conv6, training=training)
        pool6 = tf.layers.max_pooling2d(inputs=conv6_norm, pool_size=[2, 2], strides=2)
        reg_pool6 = tf.layers.dropout(inputs=pool6, rate=0.4, training=training)
        reg_pool6_flat = tf.layers.flatten(reg_pool6)
        dense = tf.layers.dense(inputs=reg_pool6_flat, 
                                    units=128, 
                                    activation=tf.nn.relu,
                                    kernel_initializer=tf.initializers.he_normal())
        dense_norm = tf.layers.batch_normalization(dense, training=training)
        reg_dense_norm = tf.layers.dropout(inputs=dense_norm, rate=0.5, training=training)
        outputs = tf.layers.dense(inputs=reg_dense_norm,
                                      units=10, 
                                      activation=None,
                                      kernel_initializer=tf.initializers.he_normal())
        return outputs

def softmax_num_stable(logits):
    logits = logits - tf.reduce_max(logits, axis=0, keepdims=True)
    return tf.exp(logits) / tf.reduce_sum(tf.exp(logits), axis=0, keepdims=True)

def train(n_epochs, learning_rate, batch_size):
    raw_train_data, raw_test_data = cifar10.load_data()
    #set up the training routines
    train_dataset = DataLoader(raw_train_data, batch_size, is_training=True, augment=True)
    X, y = train_dataset.next_batch
    y_outs = build_model(X, reuse=False, training=True)
    y_pred = softmax_num_stable(y_outs)
    train_loss = tf.losses.softmax_cross_entropy(y, y_pred)
    train_acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y, 1), tf.argmax(y_pred, 1)), tf.float32))
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(train_loss)
    #set up the testing routines
    test_dataset = DataLoader(raw_test_data, batch_size, is_training=False, augment=False)
    X_test, y_test = test_dataset.next_batch
    y_outs_test = build_model(X_test, reuse=True, training=False)
    y_pred_test = softmax_num_stable(y_outs_test)
    test_loss = tf.losses.softmax_cross_entropy(y_test, y_pred_test)
    test_acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_test, 1),tf.argmax(y_pred_test, 1)), tf.float32))
    # start the learning and evaluation process
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(n_epochs):
            train_dataset.init(sess)
            epoch_errs = []
            epoch_accs = []
            try:
                while True:
                    _, err, acc = sess.run([train_op, train_loss, train_acc])
                    epoch_errs.append(err)
                    epoch_accs.append(acc)
            except tf.errors.OutOfRangeError:
                print(f"Epoch {epoch}:")
                print(f"AVG Loss on training data: {np.mean(epoch_errs)}")
                print(f"AVG Accuracy on training data: {np.mean(epoch_accs)}")
                test_dataset.init(sess)
                test_errs = []
                test_accs = []
                try:
                    while True:
                        err, acc = sess.run([test_loss, test_acc])
                        test_errs.append(err)
                        test_accs.append(acc)
                except tf.errors.OutOfRangeError:
                    epoch_test_err, epoch_test_acc = np.mean(test_errs), np.mean(test_accs) 
                print(f"AVG Loss on testing data: {epoch_test_err}")
                print(f"AVG Accuracy on testing data: {epoch_test_acc}")

@click.command()
@click.option('--epochs', type=int, default=20)
@click.option('--batch', type=int, default=64)
@click.option('--lr', type=float, default=1e-3)

def main(epochs, lr, batch):
    train(epochs, lr, batch)

if __name__ == '__main__':
    main()
