import tensorflow as tf
import os
from data_loaders import DataLoaderFromArrays
import models
import coverage_analyzers as cov_anal
import click

def train_lenet(N_epochs, batch_size, checkpoint_path):	
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    train_data = DataLoaderFromArrays(x_train, y_train, one_hot=True, normalization=True)
    test_data = DataLoaderFromArrays(x_test, y_test, one_hot=True, normalization=True)
    X_test, Y_test = test_data.get_data()
    model = models.LeNet()
    saver = tf.train.Saver()
    best_test_accurary = 0.0
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(N_epochs):
            for i in range(train_data.N_instances // batch_size):
                batch_x, batch_y = train_data.next_batch(batch_size)
                sess.run(model.train_op, feed_dict={model.features: batch_x, model.labels: batch_y})
                if i % 50 == 0:
                    loss, accurary = sess.run([model.loss, model.accuracy],
                                        feed_dict={model.features: batch_x, model.labels: batch_y})
                    print('[Epoch {}] i: {} Loss: {} Accurary: {}'.format(epoch, i, loss, accurary))
            test_accurary = sess.run(model.accuracy, 
                                    feed_dict={model.features: X_test, model.labels: Y_test})
            print('Test Accurary: {}'.format(test_accurary))
            if best_test_accurary < test_accurary:
                saver.save(sess, checkpoint_path)
                best_test_accurary = test_accurary
            print('Best Test Accurary: {}'.format(best_test_accurary))

@click.command()
@click.option('--epochs', type=int, default=5)
@click.option('--batch', type=int, default=64)
def main(epochs, batch):
    if not os.path.isdir('./backup'):
        os.mkdir('./backup')
    checkpoint_path = "./backup/lenet.ckpt"
    train_lenet(epochs, batch, checkpoint_path)

if __name__ == '__main__':
    main()