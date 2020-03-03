# Hands-On Coding SEMLA-IVADO: Computer-Vision Model Testing
The objective is to implement a metamorphic testing solution for detecting potential issues of Deep Neural Networks (DNNs) applied in computer vision. To enable the automation of large-scale test cases generation, it should leverage multiple image transformations to build large number of metamorphic transformations and their following-up tests, with aim of finding DNN’s erroneous behaviors. In fact, the defined metamorphic transformation need to be designed to preserve both of transformed and genuine input semantically equivalent. Thus, we should respect some conservative rules and sanitize the validity of each mutated input data before checking the following-up test. As an indicator of the amount of logic explored by testing inputs, we use neuronal coverage criteria that, conceptually, estimate the coverage of neurons' activations states. Therefore, we should store each valid mutant created that was capable to spawn a failed test.

**Notice: No prior knowledge on image processing and Tensorflow (TF) development is required. All the code related to the TF models and image processing is given and your coding tasks focus on metamorphic testing and coverage evaluation.**

## Metamorphic Transformation
The component `transformer` contains different mutations from two categories of image-based transformations:
1. Pixel value transformations: change image contrast, image brightness, image blur, image sharpness and random perturbations within a valid range.
2. Affine transformations: image translation, image scaling, image shearing and image rotation.

The first task is to implement `transformer.apply_random_transformation` that applies a metamorphic transformation on the input, assembling the provided image-based transformations to guarantee the diversity of generated inputs. It applies sequentially the pixel-value transformations with respect to the given valid range in metadata, i.e. high and low bounds for each transformation’s parameter. Then, it computes the structural similarity between the transformed image and its original version (use `skimage.measure.compare_ssim`). Indeed, we adopt a conservative strategy that reject the mutated images, if its computed similarity w.r.t original input is higher than a pre-fixed threshold in order to reduce this risk of meaningless inputs. Last, it selects a single affine transformation for each generated input because applying multiple affine transformations at once could enhance the chances of generating meaningless images.

As a test, you can use the following execution that allows to sample 1000 random images from MNIST test data, perform random metamorphic transformations on them, store in _test_images_ folder the inputs whose corresponding ssim is higher than 0.75 :
```console
houssem@semla:~$ python transformer.py --max 1000 --ssim 0.75 
```
## Follow-up Test
The second task is to implement the `NC.curr_coverage` that computes the overall neurons coverage whenever it is called.
The third task is to develop the `Generator.check_adv_objective` that takes the logits returned by the DNN for the transformed data to check if the test fails or succeeds, then stores the image corresponding to the failure tests.

As a test, you can use the following execution that allows to train the variant of LeNet model and perform the metamorphic testing while providing the NC value reached by the generated test cases :
```console
houssem@semla:~$ python train.py --epochs 10 --batch 64
houssem@semla:~$ python test.py --n 100 --max 10
```
