# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 22:40:57 2020

@author: Tanmay Thakur
"""
import numpy as np

from keras import backend as K
from keras.applications import vgg19
from keras.preprocessing.image import load_img, img_to_array


# This is the path to the image you want to transform.
target_image_path = '/home/ubuntu/data/portrait.png'
# This is the path to the style image.
style_reference_image_path = '/home/ubuntu/data/popova.jpg'

# Dimensions of the generated picture.
width, height = load_img(target_image_path).size
img_height = 400
img_width = int(width * img_height / height)

def preprocess_image(image_path):
    img = load_img(image_path, target_size=(img_height, img_width))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = vgg19.preprocess_input(img)
    return img

def deprocess_image(x):
    # Remove zero-center by mean pixel
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    # 'BGR'->'RGB'
    x = x[:, :, ::-1]
    x = np.clip(x, 0, 255).astype('uint8')
    return x

target_image = K.constant(preprocess_image(target_image_path))
style_reference_image = K.constant(preprocess_image(style_reference_image_path))

# This placeholder will contain our generated image
combination_image = K.placeholder((1, img_height, img_width, 3))

# We combine the 3 images into a single batch
input_tensor = K.concatenate([target_image,
                              style_reference_image,
                              combination_image], axis=0)

# We build the VGG19 network with our batch of 3 images as input.
# The model will be loaded with pre-trained ImageNet weights.
model = vgg19.VGG19(input_tensor=input_tensor,
                    weights='imagenet',
                    include_top=False)
print('Model loaded.')

# Let's define the content loss, meant to make sure that the top layer of the VGG19 convnet will have a similar view of the target image and the generated image:

def content_loss(base, combination):
    return K.sum(K.square(combination - base))

"""
Now, here's the style loss. 
It leverages an auxiliary function to compute the Gram matrix of an input matrix, i.e. a map of the correlations found in the original feature matrix.
"""

def gram_matrix(x):
    features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
    gram = K.dot(features, K.transpose(features))
    return gram


def style_loss(style, combination):
    S = gram_matrix(style)
    C = gram_matrix(combination)
    channels = 3
    size = img_height * img_width
    return K.sum(K.square(S - C)) / (4. * (channels ** 2) * (size ** 2))


"""

To these two loss components, we add a third one, the "total variation loss". 
It is meant to encourage spatial continuity in the generated image, thus avoiding overly pixelated results. 
You could interpret it as a regularization loss.

"""

def total_variation_loss(x):
    a = K.square(
        x[:, :img_height - 1, :img_width - 1, :] - x[:, 1:, :img_width - 1, :])
    b = K.square(
        x[:, :img_height - 1, :img_width - 1, :] - x[:, :img_height - 1, 1:, :])
    return K.sum(K.pow(a + b, 1.25))

"""
The loss that we minimize is a weighted average of these three losses. 
To compute the content loss, we only leverage one top layer, the block5_conv2 layer, while for the style loss we use a list of layers than spans both low-level and high-level layers. 
We add the total variation loss at the end.

Depending on the style reference image and content image you are using, you will likely want to tune the content_weight coefficient, the contribution of the content loss to the total loss. 
A higher content_weight means that the target content will be more recognizable in the generated image.
"""

# Dict mapping layer names to activation tensors
outputs_dict = dict([(layer.name, layer.output) for layer in model.layers])
# Name of layer used for content loss
content_layer = 'block5_conv2'
# Name of layers used for style loss
style_layers = ['block1_conv1',
                'block2_conv1',
                'block3_conv1',
                'block4_conv1',
                'block5_conv1']
# Weights in the weighted average of the loss components
total_variation_weight = 1e-4
style_weight = 1.
content_weight = 0.025

# Define the loss by adding all components to a `loss` variable
loss = K.variable(0.)
layer_features = outputs_dict[content_layer]
target_image_features = layer_features[0, :, :, :]
combination_features = layer_features[2, :, :, :]
loss += content_weight * content_loss(target_image_features,
                                      combination_features)
for layer_name in style_layers:
    layer_features = outputs_dict[layer_name]
    style_reference_features = layer_features[1, :, :, :]
    combination_features = layer_features[2, :, :, :]
    sl = style_loss(style_reference_features, combination_features)
    loss += (style_weight / len(style_layers)) * sl
loss += total_variation_weight * total_variation_loss(combination_image)

"""
Finally, we set up the gradient descent process. 
In the original Gatys et al. paper, optimization is performed using the L-BFGS algorithm, so that is also what we will use here. 
This is a key difference from the Deep Dream example in the previous section. The L-BFGS algorithms comes packaged with SciPy. 
However, there are two slight limitations with the SciPy implementation:

It requires to be passed the value of the loss function and the value of the gradients as two separate functions.
It can only be applied to flat vectors, whereas we have a 3D image array.
It would be very inefficient for us to compute the value of the loss function and the value of gradients independently, since it would lead to a lot of redundant computation between the two. 
We would be almost twice slower than we could be by computing them jointly. 
To by-pass this, we set up a Python class named Evaluator that will compute both loss value and gradients value at once, will return the loss value when called the first time, and will cache the gradients for the next call.
"""

# Get the gradients of the generated image wrt the loss
grads = K.gradients(loss, combination_image)[0]

# Function to fetch the values of the current loss and the current gradients
fetch_loss_and_grads = K.function([combination_image], [loss, grads])


class Evaluator(object):

    def __init__(self):
        self.loss_value = None
        self.grads_values = None

    def loss(self, x):
        assert self.loss_value is None
        x = x.reshape((1, img_height, img_width, 3))
        outs = fetch_loss_and_grads([x])
        loss_value = outs[0]
        grad_values = outs[1].flatten().astype('float64')
        self.loss_value = loss_value
        self.grad_values = grad_values
        return self.loss_value

    def grads(self, x):
        assert self.loss_value is not None
        grad_values = np.copy(self.grad_values)
        self.loss_value = None
        self.grad_values = None
        return grad_values

evaluator = Evaluator()