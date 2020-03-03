# Hands-On Coding SEMLA-IVADO: Debugging A TF-based training program

The goal of this hands-on session is to practice troubleshooting a deep neural network implementation.
This is a VGG-like ConvNet trained on CIFAR-10 classficiation. If you succeed to fix the bugs (that I know), 
it should achieve more than 80% accuracy on the test set after more than 20 epochs.
The debugging process is more important than finding the bug. You should be able to describe the steps you've done towards hunting the bugs and it is not about doing an in-depth review of code based on your Tensorflow / Neural Network coding skills.

## Some TF routines that you may use in your debugging:
### Take a sample of the data:
```python
dset = dset.take(sample_size)
```
### Get all/trainable variables of the graph
```python
var_names = [var.name for var in tf.all_variables()]
var_nodes = {var_name:_as_graph_element(var_name) for var_name in var_names}
var_arrays = session.run(var_nodes)
```
### Compute the gradient of function w.r.t arguments:
```python
grad_f_x = tf.gradients(f, x)
grad_f_x_array = session.run(grad_f_x)    
```
### Store an image from a numpy array
```python
from PIL import Image
im = Image.fromarray(image)
im.save(file_name)
```





