import click
from data_loaders import DataLoaderFromArrays
import tensorflow as tf
import numpy as np
import models
import generators
import coverage_analyzers as cov_anal

def test_lenet(checkpoint_path, model, n_elements, max_iterations, testing_dataset, coverage_analyzer):
    '''
    perform a metamorphic testing to TF-based model
    :param checkpoint_path: the path of checkpoint storing the model's parameters
    :param model: the model under test
    :param n_elements: number of elements sampled from the original test dataset
    :param max_iterations: max number of test cases generated from one original test input
    :param testing_dataset: the original test data
    :param coverage_analyzer: this component ensures measuring the coverage criteria 
    '''
    x_test, y_test = testing_dataset
    test_data = DataLoaderFromArrays(x_test, y_test, normalization=False, one_hot=False)
    sample_features, sample_labels = test_data.next_batch(n_elements)
    prep_sample_features = test_data.normalize(sample_features)
    prep_sample_labels = test_data.one_hot_encoding(sample_labels)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, checkpoint_path)
        tensors = coverage_analyzer.get_tensors()
        outputs  = sess.run([model.accuracy, model.correct_prediction]+tensors, feed_dict={model.features: prep_sample_features, model.labels: prep_sample_labels})
        test_accurary = outputs[0]
        test_prediction = outputs[1]
        tensors_values = outputs[2:]
        coverage_analyzer.update_coverage(tensors_values)
        curr_coverage = coverage_analyzer.curr_coverage() 
        print('initial coverage: {}'.format(curr_coverage))
        print('test accurary: {}'.format(test_accurary))
        sample_features = sample_features[test_prediction]
        sample_labels = sample_labels[test_prediction]
        sample_size = np.sum(test_prediction)
        generator = generators.Generator(sess, model, coverage_analyzer)
        for i in range(sample_size):
            print('running the example {}/{}'.format(i,sample_size))
            image = sample_features[i]
            curr_target = sample_labels[i]
            generator.run(image, curr_target, max_iterations)

@click.command()
@click.option('--n', type=int, default=50)
@click.option('--max', type=int, default=10)
def main(n, max):
    _, raw_testing_data = tf.keras.datasets.mnist.load_data()
    model = models.LeNet()
    activations = ['LeNet/conv2d/Relu:0', 'LeNet/conv2d_1/Relu:0', 'LeNet/dense/Relu:0']
    coverage_analyzer = cov_anal.NC(activations)
    checkpoint_path = "./backup/lenet.ckpt"
    test_lenet(checkpoint_path=checkpoint_path, 
               model=model, 
               n_elements=n, 
               max_iterations=max, 
               testing_dataset=raw_testing_data, 
               coverage_analyzer=coverage_analyzer)

if __name__ == '__main__':
    main()