import tensorflow as tf

class LeNet:

    def __init__(self):
        self.features = tf.placeholder(tf.float32, [None, 28, 28])
        self.images = tf.reshape(self.features, [-1, 28, 28, 1])
        self.labels = tf.placeholder(tf.float32, [None, 10])
        self.logits = self.build(self.images)
        self.probabilities = tf.nn.softmax(self.logits)
        self.correct_prediction = tf.equal(tf.argmax(self.probabilities, 1), tf.argmax(self.labels, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, "float"))
        self.reg_loss = tf.losses.get_regularization_loss()
        self.loss = tf.losses.softmax_cross_entropy(self.labels, self.logits)
        self.train_op = tf.train.AdamOptimizer(3e-4).minimize((self.loss+self.reg_loss))

    def build(self, features, l1=0.0, l2=5e-4):
        with tf.variable_scope('LeNet'):
            # Convolution Layer with 32 filters and a kernel size of 5
            conv1 = tf.layers.conv2d(inputs=features,
                                    filters=32,
                                    kernel_size=[5, 5], 
                                    padding='same',
                                    activation=tf.nn.relu,
                                    kernel_regularizer=tf.keras.regularizers.l1_l2(l1,l2))
            # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
            pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

            # Convolution Layer with 64 filters and a kernel size of 3
            conv2 = tf.layers.conv2d(inputs=pool1, 
                                    filters=64, 
                                    kernel_size=[3, 3],
                                    padding='same', 
                                    activation=tf.nn.relu,
                                    kernel_regularizer=tf.keras.regularizers.l1_l2(l1,l2))
            # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
            pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

            # Flatten the data to a 1-D vector for the fully connected layer
            pool2_flat = tf.layers.flatten(pool2)

            # Fully connected layer 
            dense = tf.layers.dense(inputs=pool2_flat, 
                                    units=1024, 
                                    activation=tf.nn.relu,
                                    kernel_regularizer=tf.keras.regularizers.l1_l2(l1,l2))

            outputs = tf.layers.dense(inputs=dense,
                                      units=10, 
                                      activation=None,
                                      kernel_regularizer=tf.keras.regularizers.l1_l2(l1,l2))
        return outputs