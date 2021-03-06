import math
import os
import time
from math import ceil
import multiprocessing

import cv2
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_nn_ops

from imgaug_ import augmenters as iaa
from imgaug_ import imgaug_1
from libs.activations import lrelu
from libs.utils import corrupt
from conv2d import Conv2d
from max_pool_2d import MaxPool2d
import datetime
import io
from PIL import Image
from skimage import io as skio
from skimage.exposure import equalize_adapthist, adjust_gamma
from sklearn.model_selection import KFold, cross_val_score
import random
from sklearn.metrics import precision_recall_fscore_support, cohen_kappa_score, roc_auc_score, confusion_matrix, roc_curve, auc as auc_
from PIL import Image
from random import randint

IMAGE_HEIGHT = 565
# IMAGE_HEIGHT = 584
IMAGE_WIDTH = 584
# INPUT_IMAGE_HEIGHT = 600
# INPUT_IMAGE_WIDTH = 600
INPUT_IMAGE_HEIGHT = IMAGE_HEIGHT
INPUT_IMAGE_WIDTH = IMAGE_WIDTH

Mod_HEIGHT = 584
Mod_WIDTH = 584

# np.set_printoptions(threshold=np.nan)

"""
@ops.RegisterGradient("MaxPoolWithArgmax")
def _MaxPoolWithArgmaxGrad(op, grad, unused_argmax_grad):
    return gen_nn_ops._max_pool_grad(op.inputs[0],
                                     op.outputs[0],
                                     grad,
                                     op.get_attr("ksize"),
                                     op.get_attr("strides"),
                                     padding=op.get_attr("padding"),
                                     data_format='NHWC')
"""

#assume this is square
def tile_images(list_of_images, new_shape = (4000, 4000), save=True, file_name = "layer1_collage.jpeg"):
    list_of_images.sort(key=lambda x: x[0])
    num_images = len(list_of_images)
    sqrt_num_images = int(num_images**(.5))
    new_image = np.zeros(new_shape)
    resize_height = int(new_shape[0] / sqrt_num_images)
    resize_width = int(new_shape[1] / sqrt_num_images)
    grid_positions = []

    for i in range(num_images):
        list_of_images[i] = cv2.resize(list_of_images[i][1], (resize_height, resize_width), cv2.INTER_AREA)

    for i in range(sqrt_num_images):
        for j in range(sqrt_num_images):
            grid_positions += [np.array((i,j))]

    for i in range(num_images):
        pos, grid_positions = find_closest_pos(grid_positions)
        x_index = resize_height*pos[0]
        y_index = resize_width*pos[1]
        new_image[x_index:x_index+resize_height, y_index:y_index+resize_width] = list_of_images[i]
    if save:
        plt.imsave(file_name, new_image)
    return new_image


def find_closest_pos(positions, start_pos=(0,0)):
    min = np.inf
    min_index = -1
    start_pos = np.array(start_pos)
    for i in range(len(positions)):
        pos = positions[i]
        test = np.linalg.norm(start_pos - pos)
        if test < min:
            min = test
            min_index = i
    min_pos = positions.pop(min_index)
    return min_pos, positions


def mask_op_and_mask_mean(correct_pred, mask, num_batches=1, width=IMAGE_WIDTH, height=IMAGE_HEIGHT):
    correct_pred = tf.multiply(correct_pred, mask)
    return mask_mean(tf.count_nonzero(correct_pred, dtype=tf.float32), mask, num_batches, width, height)


def mask_op_and_mask_mean_diff(correct_pred, mask, num_batches=1, width=IMAGE_WIDTH, height=IMAGE_HEIGHT):
    correct_pred = tf.multiply(correct_pred, mask)
    return mask_mean(tf.count_nonzero(correct_pred, dtype=tf.float32), mask, num_batches, width, height)


def mask_mean(masked_pred, mask, num_batches=1, width=IMAGE_WIDTH, height=IMAGE_HEIGHT):
    ones = tf.ones([num_batches, width, height, 1], tf.float32)
    FOV_num_pixels = tf.count_nonzero(tf.cast(tf.equal(mask, ones), tf.float32), dtype=tf.float32)
    return tf.divide(masked_pred, FOV_num_pixels)


def mask_mean_diff(masked_pred, mask, num_batches=1, width=IMAGE_WIDTH, height=IMAGE_HEIGHT):
    ones = tf.ones([num_batches, width, height], tf.float32)
    FOV_num_pixels = tf.count_nonzero(tf.cast(tf.equal(mask, ones), tf.float32), dtype=tf.float32)
    return tf.divide(masked_pred, FOV_num_pixels)


def find_class_balance(targets, masks):
    total_pos = 0
    total_num_pixels = 0
    for target, mask in zip(targets, masks):
        target = np.multiply(target, mask)
        total_pos += np.count_nonzero(target)
        total_num_pixels += np.count_nonzero(mask)
    total_neg = total_num_pixels - total_pos
    weight = total_neg / total_pos
    return weight, float(total_neg)/float(total_num_pixels), float(total_pos)/float(total_num_pixels)


def dice_coe(output, target, mask=None, num_batches=1, loss_type='jaccard', axis=None, smooth=1e-5):
    if mask != None:
        output = tf.multiply(output, mask)
        target = tf.multiply(target, mask)
    inse = tf.reduce_sum(output * target, axis=axis)
    if loss_type == 'jaccard':
        l = tf.reduce_sum(output * output, axis=axis)
        r = tf.reduce_sum(target * target, axis=axis)
    elif loss_type == 'sorensen':
        l = tf.reduce_sum(output, axis=axis)
        r = tf.reduce_sum(target, axis=axis)
    else:
        raise Exception("Unknow loss_type")
    ## old axis=[0,1,2,3]
    # dice = 2 * (inse) / (l + r)
    # epsilon = 1e-5
    # dice = tf.clip_by_value(dice, 0, 1.0-epsilon) # if all empty, dice = 1
    ## new haodong
    dice = (2. * inse + smooth) / (l + r + smooth)
    if mask != None:
        dice = mask_mean_diff(dice, mask, num_batches)
    else:
        dice = tf.reduce_mean(dice)
    return dice


def dice_hard_coe(output, target, mask=None, num_batches=1, threshold=0.5, axis=None, smooth=1e-5):
    output = tf.cast(output > threshold, dtype=tf.float32)
    target = tf.cast(target > threshold, dtype=tf.float32)
    if mask != None:
        output = tf.multiply(output, mask)
        target = tf.multiply(target, mask)
    inse = tf.reduce_sum(tf.multiply(output, target), axis=axis)
    l = tf.reduce_sum(output, axis=axis)
    r = tf.reduce_sum(target, axis=axis)
    ## old axis=[0,1,2,3]
    # hard_dice = 2 * (inse) / (l + r)
    # epsilon = 1e-5
    # hard_dice = tf.clip_by_value(hard_dice, 0, 1.0-epsilon)
    ## new haodong
    hard_dice = (2. * inse + smooth) / (l + r + smooth)
    ##
    if mask != None:
        hard_dice = mask_mean_diff(hard_dice, mask, num_batches)
    else:
        hard_dice = tf.reduce_mean(hard_dice)
    return hard_dice


def iou_coe(output, target, mask=None, num_batches=1, threshold=0.5, axis=None, smooth=1e-5):
    pre = tf.cast(output > threshold, dtype=tf.float32)
    truth = tf.cast(target > threshold, dtype=tf.float32)
    if mask != None:
        output = tf.multiply(output, mask)
        target = tf.multiply(target, mask)
    inse = tf.reduce_sum(tf.multiply(pre, truth), axis=axis)  # AND
    union = tf.reduce_sum(tf.cast(tf.add(pre, truth) >= 1, dtype=tf.float32), axis=axis)  # OR
    ## old axis=[0,1,2,3]
    # epsilon = 1e-5
    # batch_iou = inse / (union + epsilon)
    ## new haodong
    batch_iou = (inse + smooth) / (union + smooth)
    if mask != None:
        iou = mask_mean_diff(batch_iou, mask, num_batches)
    else:
        iou = tf.reduce_mean(batch_iou)
    return iou  # , pre, truth, inse, union

def preprocessing(img, **kwargs):
    img = dataset_normalized(img)
    img = clahe_equalized(img)
    img = adjust_gamma(img, 1.2)
    return img

def histo_equalized(img):
    img_equalized = np.empty(img.shape)
    img_equalized[0] = cv2.equalizeHist(np.array(img[0], dtype = np.uint8))
    return img_equalized

def clahe_equalized(img):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img_equalized = np.empty(img.shape)
    img_equalized = clahe.apply(np.array(img, dtype = np.uint8))
    return img_equalized

def dataset_normalized(img):
    img_normalized = np.empty(img.shape)
    img_std = np.std(img)
    img_mean = np.mean(img)
    img_normalized = (img - img_mean) / img_std
    img_normalized = ((img_normalized - np.min(img_normalized)) / (np.max(img_normalized)-np.min(img_normalized)))*255
    return img_normalized

def adjust_gamma(img, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    # apply gamma correction using the lookup table
    new_img = np.empty(img.shape)
    new_img = cv2.LUT(np.array(img, dtype = np.uint8), table)
    return new_img

class Network:
    # IMAGE_HEIGHT = 565
    IMAGE_HEIGHT = IMAGE_HEIGHT
    IMAGE_WIDTH = IMAGE_WIDTH
    # INPUT_IMAGE_HEIGHT = 600
    # INPUT_IMAGE_WIDTH = 600

    IMAGE_CHANNELS = 1

    def __init__(self, net_id, weight=1, layers=None, per_image_standardization=True, batch_norm=True,
                 skip_connections=True):
        # Define network - ENCODER (decoder will be symmetric).

        if layers == None:

            layers = []
            layers.append(Conv2d(kernel_size=3, output_channels=64, name='conv_1_1', net_id=net_id))
            # layers.append(Conv2d(kernel_size=3, strides=[1, 1, 1, 1], output_channels=64, name='conv_1_2', net_id = net_id))
            layers.append(MaxPool2d(kernel_size=2, name='max_1', skip_connection=True and skip_connections))

            layers.append(Conv2d(kernel_size=3, output_channels=128, name='conv_2_1', net_id=net_id))
            # layers.append(Conv2d(kernel_size=3, strides=[1, 1, 1, 1], output_channels=128, name='conv_2_2'))

            layers.append(MaxPool2d(kernel_size=2, name='max_2', skip_connection=True and skip_connections))
            layers.append(Conv2d(kernel_size=3, output_channels=256, name='conv_3_1', net_id=net_id))
            layers.append(Conv2d(kernel_size=3, dilation=2, output_channels=256, name='conv_3_2', net_id=net_id))

            # layers.append(Conv2d(kernel_size=3, strides=[1, 1, 1, 1], output_channels=256, name='conv_3_3'))

            layers.append(MaxPool2d(kernel_size=2, name='max_3', skip_connection=True and skip_connections))

            # layers.append(Conv2d(kernel_size=3,  output_channels=512, name='conv_4_1', net_id = net_id))
            # layers.append(Conv2d(kernel_size=3, output_channels=512, name='conv_4_2', net_id = net_id))

            # layers.append(Conv2d(kernel_size=3, strides=[1, 1, 1, 1], output_channels=512, name='conv_4_3'))

            # layers.append(MaxPool2d(kernel_size=2, name='max_4', skip_connection=True and skip_connections))

            # layers.append(Conv2d(kernel_size=3, output_channels=512, name='conv_5_1', net_id = net_id))
            # layers.append(Conv2d(kernel_size=3, output_channels=512, name='conv_5_2', net_id = net_id))
            # layers.append(Conv2d(kernel_size=3, strides=[1, 1, 1, 1], output_channels=512, name='conv_5_3'))

            # layers.append(MaxPool2d(kernel_size=2, name='max_5', skip_connection=True and skip_connections))

            layers.append(Conv2d(kernel_size=7, output_channels=4096, name='conv_6_1', net_id=net_id))
            layers.append(Conv2d(kernel_size=1, output_channels=4096, name='conv_6_2', net_id=net_id))
            # layers.append(Conv2d(kernel_size=1, strides=[1, 1, 1, 1], output_channels=1000, name='conv_6_3'))
            # self.inputs = tf.placeholder(tf.float32, [None, self.IMAGE_WIDTH, self.IMAGE_HEIGHT, self.IMAGE_CHANNELS],name='inputs')

        self.debug1 = tf.placeholder(tf.float32, [None, Mod_WIDTH, Mod_HEIGHT, self.IMAGE_CHANNELS], name='debug1')
        self.debug2 = tf.placeholder(tf.float32, [None, Mod_WIDTH, Mod_HEIGHT, self.IMAGE_CHANNELS], name='debug2')

        self.inputs = tf.placeholder(tf.float32, [None, Mod_WIDTH, Mod_HEIGHT, self.IMAGE_CHANNELS], name='inputs')
        self.masks = tf.placeholder(tf.float32, [None, self.IMAGE_WIDTH, self.IMAGE_HEIGHT, 1], name='masks')
        self.targets = tf.placeholder(tf.float32, [None, self.IMAGE_WIDTH, self.IMAGE_HEIGHT, 1], name='targets')
        self.is_training = tf.placeholder_with_default(False, [], name='is_training')

        self.layer_output1 = tf.placeholder(tf.float32, [None, None, None, 1], name='layer_outputs')
        self.layer_output2 = tf.placeholder(tf.float32, [None, None, None, 1], name='layer_outputs')
        self.layer_output3 = tf.placeholder(tf.float32, [None, None, None, 1], name='layer_outputs')
        self.layer_output4 = tf.placeholder(tf.float32, [None, None, None, 1], name='layer_outputs')
        self.layer_output5 = tf.placeholder(tf.float32, [None, None, None, 1], name='layer_outputs')
        self.layer_output6 = tf.placeholder(tf.float32, [None, None, None, 1], name='layer_outputs')
        self.layer_output7 = tf.placeholder(tf.float32, [None, None, None, 1], name='layer_outputs')
        self.layer_output8 = tf.placeholder(tf.float32, [None, None, None, 1], name='layer_outputs')
        self.layer_output9 = tf.placeholder(tf.float32, [None, None, None, 1], name='layer_outputs')

        self.layer_output10 = tf.placeholder(tf.float32, [None, None, None, 1], name='layer_outputs')
        self.layer_output11 = tf.placeholder(tf.float32, [None, None, None, 1], name='layer_outputs')
        self.layer_output12 = tf.placeholder(tf.float32, [None, None, None, 1], name='layer_outputs')
        self.layer_output13 = tf.placeholder(tf.float32, [None, None, None, 1], name='layer_outputs')
        self.layer_output14 = tf.placeholder(tf.float32, [None, None, None, 1], name='layer_outputs')
        self.layer_output15 = tf.placeholder(tf.float32, [None, None, None, 1], name='layer_outputs')
        self.layer_output16 = tf.placeholder(tf.float32, [None, None, None, 1], name='layer_outputs')
        self.layer_output17 = tf.placeholder(tf.float32, [None, None, None, 1], name='layer_outputs')
        self.layer_output18 = tf.placeholder(tf.float32, [None, None, None, 1], name='layer_outputs')

        # has to change for multiple batches
        # self.ones = tf.ones([1, self.IMAGE_WIDTH, self.IMAGE_HEIGHT], tf.int32)

        self.description = ""

        self.layers = {}

        self.debug1 = self.inputs
        #you can remove these image preprocessing techniques
        """
        if per_image_standardization:
            list_of_images_norm = tf.map_fn(tf.image.per_image_standardization, self.inputs)
            net = tf.stack(list_of_images_norm)
            self.debug2 = net
        else:
            net = self.inputs
            self.debug2 = net
        """
        net = self.inputs
        self.debug2 = net


        # ENCODER
        for i in range(len(layers)):
            layer = layers[i]
            self.layers[layer.name] = net = layer.create_layer(net)
            self.description += "{}".format(layer.get_description())
            if i == 0:
                self.layer_output1 = net
            elif i == 1:
                self.layer_output2 = net
            elif i == 2:
                self.layer_output3 = net
            elif i == 3:
                self.layer_output4 = net
            elif i == 4:
                self.layer_output5 = net
            elif i == 5:
                self.layer_output6 = net
            elif i == 6:
                self.layer_output7 = net
            elif i == 7:
                self.layer_output8 = net
            elif i == 8:
                self.layer_output9 = net

        print("Number of layers: ", len(layers))
        print("Current input shape: ", net.get_shape())

        layers.reverse()

        # DECODER
        for i in range(len(layers)):
            layer = layers[i]
            net = layer.create_layer_reversed(net, prev_layer=self.layers[layer.name])
            if i == 0:
                self.layer_output10 = net
            elif i == 1:
                self.layer_output11 = net
            elif i == 2:
                self.layer_output12 = net
            elif i == 3:
                self.layer_output13 = net
            elif i == 4:
                self.layer_output14= net
            elif i == 5:
                self.layer_output15 = net
            elif i == 6:
                self.layer_output16 = net
            elif i == 7:
                self.layer_output17 = net
            elif i == 8:
                self.layer_output18 = net

        net = tf.image.resize_image_with_crop_or_pad(net, IMAGE_WIDTH, IMAGE_HEIGHT)
        net = tf.multiply(net, self.masks)
        self.segmentation_result = tf.sigmoid(net)

        self.targets = tf.multiply(self.targets, self.masks)

        print('segmentation_result.shape: {}, targets.shape: {}'.format(self.segmentation_result.get_shape(),
                                                                        self.targets.get_shape()))

        self.cost_unweighted = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(self.targets, net, pos_weight=1))

        self.cost = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(self.targets, net, pos_weight=weight))
        print('net.shape: {}'.format(net.get_shape()))
        # = tf.sqrt(tf.reduce_mean(tf.square(self.segmentation_result - self.targets)))

        # override the methods called by minimize to debug the error
        # debug first layer to figure out what's going on

        self.train_op = tf.train.AdamOptimizer().minimize(self.cost)
        with tf.name_scope('accuracy'):
            # t_shape_list = (self.segmentation_result).get_shape().as_list()
            # num_batches = t_shape_list[0]
            num_batches = 1
            argmax_probs = tf.round(self.segmentation_result)  # 0x1
            correct_pred = tf.cast(tf.equal(argmax_probs, self.targets), tf.float32)
            # correct_pred = tf.multiply(correct_pred, self.masks)
            # FOV_num_pixels = tf.cast(tf.equal(self.masks, self.ones), tf.float32)
            # self.accuracy = tf.divide(correct_pred, FOV_num_pixels)
            self.accuracy = mask_op_and_mask_mean(correct_pred, self.masks, num_batches, IMAGE_WIDTH, IMAGE_HEIGHT)
            tf.summary.scalar('accuracy', self.accuracy)

        self.summaries = tf.summary.merge_all()


class Dataset:
    def __init__(self, batch_size, folder='drive', include_hair=True, sgd = False):
        self.folder = folder
        self.batch_size = batch_size
        self.include_hair = include_hair
        self.sgd = sgd

        # train_files, validation_files, test_files = self.train_valid_test_split(os.listdir(os.path.join(folder, 'inputs')))

        self.train_inputs, self.train_masks, self.train_targets = ([], [], [])
        self.test_inputs, self.train_masks, self.test_targets = ([], [], [])

        self.pointer = 0

    def file_paths_to_images(self, folder, file_indices, file_type = "_training", manual_suffix = "_manual1.gif"):

        inputs = []
        masks = []
        targets = []

        for file_index in file_indices:
            string_num = str(file_index).zfill(2)
            orig_file = string_num +file_type +".tif"
            file = string_num + manual_suffix
            mask_file = string_num +file_type+"_mask.gif"

            input_image = os.path.join(folder, 'inputs', orig_file)
            target1_image = os.path.join(folder, 'targets1', file)
            target2_image = os.path.join(folder, 'targets2', file)

            mask_loc = os.path.join(folder, 'masks', mask_file)

            # add training image to dataset
            print(input_image)
            test_image = cv2.imread(input_image, 1)
            test_image = test_image[:, :, 1]
            #add pre-processing methods here approximately, and comment out image standardization in tensorflow
            test_image = preprocessing(test_image, gamma = 2)

            top_pad = int((Mod_HEIGHT - IMAGE_HEIGHT) / 2)
            bot_pad = (Mod_HEIGHT - IMAGE_HEIGHT) - top_pad
            left_pad = int((Mod_WIDTH - IMAGE_WIDTH) / 2)
            right_pad = (Mod_WIDTH - IMAGE_WIDTH) - left_pad
            print("before:")
            print(test_image.shape)
            test_image = cv2.copyMakeBorder(test_image, left_pad, right_pad, top_pad, bot_pad, cv2.BORDER_CONSTANT, 0)
            print("after:")
            print(test_image.shape)
            inputs.append(test_image)

            # add mask for training image to dataset
            mask = Image.open(mask_loc)
            mask_array = np.array(mask)
            mask_array = mask_array / 255
            masks.append(mask_array)

            if os.path.exists(target1_image):

                # load grayscale
                # test_image = np.multiply(test_image, 1.0 / 255)

                target1_image = np.array(skio.imread(target1_image))
                target1_image = cv2.threshold(target1_image, 127, 1, cv2.THRESH_BINARY)[1]
                # target1_image = cv2.resize(target1_image, (IMAGE_WIDTH,IMAGE_HEIGHT))
                targets.append(target1_image)

            elif os.path.exists(target2_image):

                target2_image = np.array(skio.imread(target2_image))[:, :, 3]
                target2_image = cv2.threshold(target2_image, 127, 1, cv2.THRESH_BINARY)[1]

                targets.append(target2_image)
            else:
                print(target1_image)
                print(target2_image)
                print("here")

        return np.asarray(inputs), np.asarray(masks), np.asarray(targets)

    def train_valid_test_split(self, X, ratio=None):
        if ratio is None:
            ratio = (0.5, .25, .25)

        N = len(X)
        return (
            X[:int(ceil(N * ratio[0]))],
            X[int(ceil(N * ratio[0])): int(ceil(N * ratio[0] + N * ratio[1]))],
            X[int(ceil(N * ratio[0] + N * ratio[1])):]
        )

    def num_batches_in_epoch(self):
        return int(math.floor(len(self.train_inputs) / self.batch_size))

    def reset_batch_pointer(self):
        # permutation = np.random.permutation(len(self.train_inputs))
        # self.train_inputs = [self.train_inputs[i] for i in permutation]
        # self.train_targets = [self.train_targets[i] for i in permutation]

        self.pointer = 0

    def next_batch(self):
        inputs = []
        masks = []
        targets = []
        # print(self.batch_size, self.pointer, self.train_inputs.shape, self.train_targets.shape)
        if self.sgd:
            samples = np.random.choice(len(self.train_inputs), self.batch_size)

        for i in range(self.batch_size):
            if self.sgd:
                inputs.append(np.array(self.train_inputs[samples[i]]))
                masks.append(np.array(self.train_masks[samples[i]]))
                targets.append(np.array(self.train_targets[samples[i]]))
            else:
                inputs.append(np.array(self.train_inputs[self.pointer + i]))
                masks.append(np.array(self.train_masks[self.pointer + i]))
                targets.append(np.array(self.train_targets[self.pointer + i]))

        self.pointer += self.batch_size

        return np.array(inputs, dtype=np.uint8), np.array(masks, dtype=np.uint8), np.array(targets, dtype=np.uint8)

    @property
    def test_set(self):
        return np.array(self.test_inputs, dtype=np.uint8), np.array(self.test_targets, dtype=np.uint8)


def draw_results(test_inputs, test_targets, test_segmentation, test_accuracy, network, batch_num, decision_thresh=.5):
    n_examples_to_plot = n_examples

    fig, axs = plt.subplots(4, n_examples_to_plot, figsize=(n_examples_to_plot * 3, 10), squeeze=False)
    fig.suptitle("Accuracy: {}, {}".format(test_accuracy, network.description), fontsize=20)
    for example_i in range(n_examples_to_plot):
        axs[0][example_i].imshow(test_inputs[example_i], cmap='gray')
        axs[0][example_i].axis('off')
        # print(np.sum(test_targets[example_i].astype(np.float32)))
        axs[1][example_i].imshow(test_targets[example_i].astype(np.float32), cmap='gray')
        axs[1][example_i].axis('off')
        axs[2][example_i].imshow(
            np.reshape(test_segmentation[example_i], [network.IMAGE_WIDTH, network.IMAGE_HEIGHT]),
            cmap='gray')
        axs[2][example_i].axis('off')

        test_image_thresholded = np.array(
            [0 if x < decision_thresh else 255 for x in test_segmentation[example_i].flatten()])
        axs[3][example_i].imshow(
            np.reshape(test_image_thresholded, [network.IMAGE_WIDTH, network.IMAGE_HEIGHT]),
            cmap='gray')
        axs[3][example_i].axis('off')
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)

    IMAGE_PLOT_DIR = 'image_plots/'
    if not os.path.exists(IMAGE_PLOT_DIR):
        os.makedirs(IMAGE_PLOT_DIR)

    plt.savefig('{}/figure{}.jpg'.format(IMAGE_PLOT_DIR, batch_num))
    return buf


def train(end_freq = 2000, decision_thresh = .75, score_freq=10, layer_output_freq=200, output_file="results.txt", cost_log="cost_log.txt", tuning_constant=1.0, cur_time="None"):
    BATCH_SIZE = 1
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    plt.rcParams['image.cmap'] = 'gray'

    dataset = Dataset(folder='drive', include_hair=True,
                      batch_size=BATCH_SIZE, sgd = True)

    # inputs, targets = dataset.next_batch()
    # print(inputs.shape, targets.shape)

    # augmentation_seq = iaa.Sequential([
    #     iaa.Crop(px=(0, 16)),  # crop images from each side by 0 to 16px (randomly chosen)
    #     iaa.Fliplr(0.5),  # horizontally flip 50% of the images
    #     iaa.GaussianBlur(sigma=(0, 2.0))  # blur images with a sigma of 0 to 3.0
    # ])

    augmentation_seq = iaa.Sequential([
        #iaa.Crop(px=(0, 16), name="Cropper"),  # crop images from each side by 0 to 16px (randomly chosen)
        #iaa.Fliplr(0.5, name="Flipper"),
        #iaa.GaussianBlur((0, 3.0), name="GaussianBlur"),
        #iaa.Dropout(0.02, name="Dropout"),
        #iaa.AdditiveGaussianNoise(scale=0.01 * 255, name="GaussianNoise"),
        #iaa.Affine(translate_px={"x": (-1024 // 3, 1024 // 3)}, name="Affine")
    ])

    # change the activated augmenters for binary masks,
    # we only want to execute horizontal crop, flip and affine transformation
    def activator_binmasks(images, augmenter, parents, default):
        if augmenter.name in ["GaussianBlur", "Dropout", "GaussianNoise"]:
            return False
        else:
            # default value for all other augmenters
            return default

    hooks_binmasks = imgaug_1.HooksImages(activator=activator_binmasks)

    folder = dataset.folder

    train_inputs, train_masks, train_targets = dataset.file_paths_to_images(folder, range(21, 41, 1))

    test_inputs, test_masks, test_targets = dataset.file_paths_to_images(folder, range(1,21,1), file_type="_test", manual_suffix = "_manual1.gif")

    ##DEBUG
    #pos_weight
    neg_pos_class_ratio, _, _ = find_class_balance(train_targets, train_masks)
    _, test_neg_class_frac, test_pos_class_frac  = find_class_balance(test_targets, test_masks)
    #z = 0.56
    #pos_weight = (z*neg_pos_class_ratio)/(1-z)
    #pos_weight = 1
    #tuning_constant = 1.0
    pos_weight = neg_pos_class_ratio * tuning_constant


    dataset.train_inputs = train_inputs
    dataset.train_masks = train_masks
    dataset.train_targets = train_targets
    dataset.test_inputs = test_inputs
    dataset.test_masks = test_masks
    dataset.test_targets = test_targets

    print(train_inputs.shape)
    print(train_targets.shape)
    print(train_masks.shape)
    print(test_inputs.shape)
    print(test_targets.shape)
    print(test_masks.shape)
    # test_inputs, test_targets = test_inputs[:100], test_targets[:100]
    # test_inputs = np.reshape(test_inputs, (len(test_inputs), Mod_HEIGHT, Mod_WIDTH, 1))
    test_inputs = np.reshape(test_inputs, (len(test_inputs), Mod_WIDTH, Mod_HEIGHT, 1))
    test_inputs = np.multiply(test_inputs, 1.0 / 255)

    test_targets = np.reshape(test_targets, (len(test_targets), IMAGE_WIDTH, IMAGE_HEIGHT, 1))
    test_masks = np.reshape(test_masks, (len(test_masks), IMAGE_WIDTH, IMAGE_HEIGHT, 1))

    # test_inputs = np.pad(test_inputs, ((8,8),(18,17)), 'constant', constant_values=0)
    # test_inputs=tf.image.resize_image_with_crop_or_pad(test_inputs,INPUT_IMAGE_HEIGHT,INPUT_IMAGE_WIDTH)
    # test_inputs=tf.image.resize_image_with_crop_or_pad(test_inputs,INPUT_IMAGE_HEIGHT,INPUT_IMAGE_WIDTH)


    # config = tf.ConfigProto(device_count = {'GPU': 0,'GPU': 1})

    count = 0
    with tf.device('/gpu:1'):
        # with tf.device('/cpu:0'):
        network = Network(net_id=count, weight=pos_weight)
    count += 1

    # create directory for saving model
    os.makedirs(os.path.join('save', network.description + str(count), timestamp))
    layer_output_path = os.path.join('layer_outputs', network.description + str(count), timestamp)
    os.makedirs(layer_output_path)
    layer_output_path_train = os.path.join(layer_output_path, "train")
    os.makedirs(layer_output_path_train)
    layer_output_path_test = os.path.join(layer_output_path, "test")
    os.makedirs(layer_output_path_test)

    layer_debug1_output_path_train = os.path.join(layer_output_path_train, "debug1")
    os.makedirs(layer_debug1_output_path_train)
    layer_debug2_output_path_train = os.path.join(layer_output_path_train, "debug2")
    os.makedirs(layer_debug2_output_path_train)
    layer1_output_path_train = os.path.join(layer_output_path_train, "1")
    os.makedirs(layer1_output_path_train)
    layer_mask1_output_path_train = os.path.join(layer_output_path_train, "mask1")
    os.makedirs(layer_mask1_output_path_train)
    layer2_output_path_train = os.path.join(layer_output_path_train, "2")
    os.makedirs(layer2_output_path_train)
    layer3_output_path_train = os.path.join(layer_output_path_train, "3")
    os.makedirs(layer3_output_path_train)
    layer4_output_path_train = os.path.join(layer_output_path_train, "4")
    os.makedirs(layer4_output_path_train)
    layer5_output_path_train = os.path.join(layer_output_path_train, "5")
    os.makedirs(layer5_output_path_train)
    layer6_output_path_train = os.path.join(layer_output_path_train, "6")
    os.makedirs(layer6_output_path_train)
    layer7_output_path_train = os.path.join(layer_output_path_train, "7")
    os.makedirs(layer7_output_path_train)
    layer8_output_path_train = os.path.join(layer_output_path_train, "8")
    os.makedirs(layer8_output_path_train)
    layer9_output_path_train = os.path.join(layer_output_path_train, "9")
    os.makedirs(layer9_output_path_train)
    layer10_output_path_train = os.path.join(layer_output_path_train, "10")
    os.makedirs(layer10_output_path_train)
    layer11_output_path_train = os.path.join(layer_output_path_train, "11")
    os.makedirs(layer11_output_path_train)
    layer12_output_path_train = os.path.join(layer_output_path_train, "12")
    os.makedirs(layer12_output_path_train)
    layer13_output_path_train = os.path.join(layer_output_path_train, "13")
    os.makedirs(layer13_output_path_train)
    layer14_output_path_train = os.path.join(layer_output_path_train, "14")
    os.makedirs(layer14_output_path_train)
    layer15_output_path_train = os.path.join(layer_output_path_train, "15")
    os.makedirs(layer15_output_path_train)
    layer16_output_path_train = os.path.join(layer_output_path_train, "16")
    os.makedirs(layer16_output_path_train)
    layer17_output_path_train = os.path.join(layer_output_path_train, "17")
    os.makedirs(layer17_output_path_train)
    layer_mask2_output_path_train = os.path.join(layer_output_path_train, "mask2")
    os.makedirs(layer_mask2_output_path_train)
    layer18_output_path_train = os.path.join(layer_output_path_train, "18")
    os.makedirs(layer18_output_path_train)

    layer1_output_path_test = os.path.join(layer_output_path_test, "1")
    os.makedirs(layer1_output_path_test)
    layer_mask1_output_path_test = os.path.join(layer_output_path_test, "mask1")
    os.makedirs(layer_mask1_output_path_test)
    layer2_output_path_test = os.path.join(layer_output_path_test, "2")
    os.makedirs(layer2_output_path_test)
    layer3_output_path_test = os.path.join(layer_output_path_test, "3")
    os.makedirs(layer3_output_path_test)
    layer4_output_path_test = os.path.join(layer_output_path_test, "4")
    os.makedirs(layer4_output_path_test)
    layer5_output_path_test = os.path.join(layer_output_path_test, "5")
    os.makedirs(layer5_output_path_test)
    layer6_output_path_test = os.path.join(layer_output_path_test, "6")
    os.makedirs(layer6_output_path_test)
    layer7_output_path_test = os.path.join(layer_output_path_test, "7")
    os.makedirs(layer7_output_path_test)
    layer8_output_path_test = os.path.join(layer_output_path_test, "8")
    os.makedirs(layer8_output_path_test)
    layer9_output_path_test = os.path.join(layer_output_path_test, "9")
    os.makedirs(layer9_output_path_test)
    layer10_output_path_test = os.path.join(layer_output_path_test, "10")
    os.makedirs(layer10_output_path_test)
    layer11_output_path_test = os.path.join(layer_output_path_test, "11")
    os.makedirs(layer11_output_path_test)
    layer12_output_path_test = os.path.join(layer_output_path_test, "12")
    os.makedirs(layer12_output_path_test)
    layer13_output_path_test = os.path.join(layer_output_path_test, "13")
    os.makedirs(layer13_output_path_test)
    layer14_output_path_test = os.path.join(layer_output_path_test, "14")
    os.makedirs(layer14_output_path_test)
    layer15_output_path_test = os.path.join(layer_output_path_test, "15")
    os.makedirs(layer15_output_path_test)
    layer16_output_path_test = os.path.join(layer_output_path_test, "16")
    os.makedirs(layer16_output_path_test)
    layer17_output_path_test = os.path.join(layer_output_path_test, "17")
    os.makedirs(layer17_output_path_test)
    layer_mask2_output_path_test = os.path.join(layer_output_path_test, "mask2")
    os.makedirs(layer_mask2_output_path_test)
    layer18_output_path_test = os.path.join(layer_output_path_test, "18")
    os.makedirs(layer18_output_path_test)



    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
    config.gpu_options.allow_growth = True

    saver = tf.train.Saver()

    with tf.Session(config=config) as sess:
        with tf.device('/gpu:1'):
            print(sess.run(tf.global_variables_initializer()))

            summary_writer = tf.summary.FileWriter('{}/{}-{}'.format('logs', network.description, timestamp),
                                                   graph=tf.get_default_graph())
            saver = tf.train.Saver(tf.all_variables(), max_to_keep=None)

            test_accuracies = []
            test_auc = []
            test_auc_10_fpr = []
            test_auc_05_fpr = []
            test_auc_025_fpr = []
            max_thresh_accuracies = []
            # Fit all training data
            n_epochs = 40000
            global_start = time.time()
            batch_num = 0

            #layer_output_freq = 200
            #score_freq = 1
            #end_freq = 2000
            #decision_thresh = .75
            for epoch_i in range(n_epochs):
                if batch_num > end_freq:
                    epoch_i = 0
                    dataset.reset_batch_pointer()
                    break
                dataset.reset_batch_pointer()
                for batch_i in range(dataset.num_batches_in_epoch()):
                    batch_num = epoch_i * dataset.num_batches_in_epoch() + batch_i + 1
                    if batch_num > end_freq:
                        break

                    augmentation_seq_deterministic = augmentation_seq.to_deterministic()
                    start = time.time()
                    batch_inputs, batch_masks, batch_targets = dataset.next_batch()

                    print(batch_inputs.shape)
                    plt.imsave(os.path.join(layer_output_path_train, "test1.jpeg"), batch_inputs[0])
                    plt.imsave(os.path.join(layer_output_path_train, "test1_target.jpeg"), batch_targets[0])

                    batch_inputs = np.reshape(batch_inputs, (dataset.batch_size, Mod_WIDTH, Mod_HEIGHT, 1))
                    batch_masks = np.reshape(batch_masks, (dataset.batch_size, network.IMAGE_WIDTH, network.IMAGE_HEIGHT, 1))
                    batch_targets = np.reshape(batch_targets, (dataset.batch_size, network.IMAGE_WIDTH, network.IMAGE_HEIGHT, 1))

                    batch_inputs = augmentation_seq_deterministic.augment_images(batch_inputs)
                    plt.imsave(os.path.join(layer_output_path_train, "test2.jpeg"), batch_inputs[0, :, :, 0])
                    plt.imsave(os.path.join(layer_output_path_train, "test2_target.jpeg"), batch_targets[0, :, :, 0])

                    batch_inputs = np.multiply(batch_inputs, 1.0 / 255)

                    batch_targets = augmentation_seq_deterministic.augment_images(batch_targets, hooks=hooks_binmasks)

                    cost, cost_unweighted, layer_output1, layer_output2, layer_output3, layer_output4, layer_output5, layer_output6, layer_output7, \
                    layer_output8, layer_output9, layer_output10,  layer_output11, layer_output12, layer_output13, layer_output14, layer_output15, \
                    layer_output16, layer_output17, layer_output18, debug1, debug2, _ = sess.run([network.cost, network.cost_unweighted, network.layer_output1, network.layer_output2,
                                                                       network.layer_output3, network.layer_output4, network.layer_output5,
                                                                       network.layer_output6, network.layer_output7, network.layer_output8,
                                                                       network.layer_output9, network.layer_output10, network.layer_output11,
                                                                       network.layer_output12, network.layer_output13, network.layer_output14,
                                                                       network.layer_output15, network.layer_output16, network.layer_output17,
                                                                       network.layer_output18, network.debug1, network.debug2, network.train_op],
                                       feed_dict={network.inputs: batch_inputs, network.masks: batch_masks, network.targets: batch_targets, network.is_training: True})
                    layer_outputs = [layer_output1, layer_output2, layer_output3, layer_output4, layer_output5, layer_output6,
                                     layer_output7, layer_output8, layer_output9, layer_output10, layer_output11, layer_output12,
                                     layer_output13, layer_output14, layer_output15, layer_output16, layer_output17, layer_output18]

                    end = time.time()

                    print('{}/{}, epoch: {}, cost: {}, cost unweighted: {}, batch time: {}, positive_weight: {}'.format(batch_num,
                                                                                                                        n_epochs * dataset.num_batches_in_epoch(),
                                                                                                                        epoch_i,
                                                                                                                        cost,
                                                                                                                        cost_unweighted,
                                                                                                                        end - start,
                                                                                                                       pos_weight))
                    debug1 = debug1[0,:,:,0]
                    debug2 = debug2[0,:,:,0]
                    plt.imsave(os.path.join(layer_debug1_output_path_train, "debug1.jpeg"), debug1)
                    plt.imsave(os.path.join(layer_debug2_output_path_train, "debug2.jpeg"), debug2)

                    mask_threshold = .5
                    channel_list = []

                    if batch_num % score_freq == 0 or batch_num == n_epochs * dataset.num_batches_in_epoch():
                        if batch_num % layer_output_freq == 0:
                            for j in range(len(layer_outputs)):
                                layer_output = layer_outputs[j]
                                for k in range(layer_output.shape[3]):
                                    channel_output = layer_output[0,:,:,k]
                                    plt.imsave(os.path.join(os.path.join(layer_output_path_train, str(j+1)),"channel_"+str(k)+ ".jpeg"), channel_output)
                                    if j == 0:
                                        channel_output[np.where(channel_output > mask_threshold)] = 1
                                        channel_output[np.where(channel_output <= mask_threshold)] = 0
                                        plt.imsave(os.path.join(os.path.join(layer_output_path_train, "mask1"), "channel_" + str(k) + ".jpeg"), channel_output)
                                    if j == 16:
                                        channel_output[np.where(channel_output > mask_threshold)] = 1
                                        channel_output[np.where(channel_output <= mask_threshold)] = 0
                                        plt.imsave(os.path.join(os.path.join(layer_output_path_train, "mask2"), "channel_" + str(k) + ".jpeg"), channel_output)

                        test_accuracy = 0.0
                        max_thresh_accuracy = 0.0

                        test_cost = 0.0
                        test_cost_unweighted = 0.0

                        mask_array = np.zeros((len(test_inputs), IMAGE_WIDTH, IMAGE_HEIGHT))
                        target_array = np.zeros((len(test_inputs), IMAGE_WIDTH, IMAGE_HEIGHT))
                        prediction_array = np.zeros((len(test_inputs), IMAGE_WIDTH, IMAGE_HEIGHT))

                        sample_test_image = randint(0, len(test_inputs)-1)
                        for i in range(len(test_inputs)):
                            thresh_max = 0.0
                            if i == sample_test_image and batch_num % layer_output_freq == 0:
                                test_cost_, test_cost_unweighted_, inputs, masks, results, targets, acc, layer_output1, layer_output2, layer_output3, layer_output4, layer_output5, layer_output6, layer_output7, \
                                layer_output8, layer_output9, layer_output10, layer_output11, layer_output12, layer_output13, layer_output14, layer_output15, layer_output16, \
                                layer_output17, layer_output18 = sess.run(
                                    [network.cost, network.cost_unweighted, network.inputs, network.masks, network.segmentation_result, network.targets, network.accuracy, network.layer_output1, network.layer_output2,
                                     network.layer_output3, network.layer_output4, network.layer_output5,
                                     network.layer_output6, network.layer_output7, network.layer_output8,
                                     network.layer_output9, network.layer_output10, network.layer_output11,
                                     network.layer_output12, network.layer_output13, network.layer_output14,
                                     network.layer_output15, network.layer_output16, network.layer_output17,
                                     network.layer_output18],
                                    feed_dict={network.inputs: test_inputs[i:(i + 1)], network.masks: test_masks[i:(i + 1)], network.targets: test_targets[i:(i + 1)],network.is_training: False})
                                layer_outputs = [layer_output1, layer_output2, layer_output3, layer_output4, layer_output5, layer_output6,
                                                 layer_output7, layer_output8, layer_output9, layer_output10, layer_output11, layer_output12,
                                                 layer_output13, layer_output14, layer_output15, layer_output16, layer_output17, layer_output18]

                                top_pad = int((Mod_HEIGHT - IMAGE_HEIGHT) / 2)
                                bot_pad = (Mod_HEIGHT - IMAGE_HEIGHT) - top_pad
                                left_pad = int((Mod_WIDTH - IMAGE_WIDTH) / 2)
                                right_pad = (Mod_WIDTH - IMAGE_WIDTH) - left_pad

                                target = test_targets[i:(i + 1)]
                                target = np.reshape(target, (IMAGE_HEIGHT, IMAGE_WIDTH))
                                invert_target = np.invert(target)

                                mask = test_masks[i:(i + 1)]
                                mask = np.reshape(mask, (IMAGE_HEIGHT, IMAGE_WIDTH))
                                target = np.multiply(target, mask)
                                invert_target = np.multiply(invert_target, mask)

                                #input =  test_inputs[i:(i + 1)]
                                #input = np.reshape(input, (Mod_HEIGHT, Mod_WIDTH))
                                #input = input[top_pad:Mod_HEIGHT-bot_pad,left_pad:Mod_WIDTH-right_pad]



                                channel_list = []
                                total_pos = 0
                                total_neg = 0

                                for j in range(len(layer_outputs)):
                                    layer_output = layer_outputs[j]
                                    for k in range(layer_output.shape[3]):
                                        channel_output = layer_output[0, :, :, k]
                                        plt.imsave(os.path.join(os.path.join(layer_output_path_test, str(j + 1)),
                                                              "channel_" + str(k) + ".jpeg"), channel_output)
                                        if j == 0:
                                            input = np.reshape(channel_output, (Mod_HEIGHT, Mod_WIDTH))
                                            input_crop = input[top_pad:Mod_HEIGHT - bot_pad, left_pad:Mod_WIDTH - right_pad]
                                            invert_input_crop = 1-input_crop

                                            results_ = np.multiply(input_crop, target)
                                            results_invert = np.multiply(invert_input_crop, invert_target)

                                            sum = np.sum(results_)
                                            sum_neg = np.sum(results_invert)

                                            total_pos += sum
                                            total_neg += sum_neg

                                            channel_list.append((sum, input_crop))

                                            channel_output[np.where(channel_output > mask_threshold)] = 1
                                            channel_output[np.where(channel_output <= mask_threshold)] = 0
                                            plt.imsave(os.path.join(os.path.join(layer_output_path_test, "mask1"),
                                                                    "channel_" + str(k) + ".jpeg"), channel_output)
                                        if j == 16:
                                            channel_output[np.where(channel_output > mask_threshold)] = 1
                                            channel_output[np.where(channel_output <= mask_threshold)] = 0
                                            plt.imsave(os.path.join(os.path.join(layer_output_path_test, "mask2"),
                                                                    "channel_" + str(k) + ".jpeg"),channel_output)

                            else:
                                test_cost_, test_cost_unweighted_, inputs, masks, results, targets, acc = sess.run([network.cost, network.cost_unweighted, network.inputs, network.masks, network.segmentation_result, network.targets, network.accuracy],
                                                                                feed_dict={network.inputs: test_inputs[i:(i + 1)], network.masks: test_masks[i:(i + 1)],
                                                                                           network.targets: test_targets[i:(i + 1)], network.is_training: False})
                            masks = masks[0, :, :, 0]
                            results = results[0, :, :, 0]
                            targets = targets[0, :, :, 0]

                            mask_array[i] = masks
                            target_array[i] = targets
                            prediction_array[i] = results

                            test_accuracy += acc
                            test_cost += test_cost_
                            test_cost_unweighted += test_cost_unweighted_

                            fprs, tprs, thresholds = roc_curve(targets.flatten(), results.flatten(), sample_weight=masks.flatten())
                            list_fprs_tprs_thresholds = list(zip(fprs, tprs, thresholds))
                            interval = 0.0001

                            for i in np.arange(0.0, 1.0 + interval, interval):
                                index = int(round((len(thresholds) - 1) * i, 0))
                                fpr, tpr, threshold = list_fprs_tprs_thresholds[index]
                                thresh_acc = (1 - fpr) * test_neg_class_frac + tpr * test_pos_class_frac
                                if thresh_acc > thresh_max:
                                    thresh_max = thresh_acc
                                i += 1

                            max_thresh_accuracy += thresh_max

                        #print("print positive correlation sums")
                        #for it in channel_list:
                        #    print(it[0])
                        """
                        """

                        max_thresh_accuracy = max_thresh_accuracy / len(test_inputs)
                        test_accuracy = test_accuracy / len(test_inputs)
                        test_cost = test_cost / len(test_inputs)
                        test_cost_unweighted = test_cost_unweighted / len(test_inputs)

                        #mask_tensor = tf.convert_to_tensor(mask_array, dtype=tf.float32)
                        #prediction_tensor = tf.convert_to_tensor(prediction_array, dtype=tf.float32)
                        #target_tensor = tf.convert_to_tensor(target_array, dtype=tf.float32)

                        #dice_coe_val = dice_coe(prediction_tensor, target_tensor, mask_tensor, len(test_inputs))
                        #hard_dice_coe_val = dice_hard_coe(prediction_tensor, target_tensor, mask_tensor,len(test_inputs))
                        #iou_coe_val = iou_coe(prediction_tensor, target_tensor, mask_tensor, len(test_inputs))

                        mask_flat = mask_array.flatten()
                        prediction_flat = prediction_array.flatten()
                        target_flat = target_array.flatten()

                        ## save mask and target (if files don't exist)
                        cwd = os.getcwd()
                        if not os.path.exists(os.path.join(cwd, "mask.npy")):
                            np.save("mask", mask_flat)
                        if not os.path.exists(os.path.join(cwd, "target.npy")):
                            np.save("target", target_flat)

                        ### save prediction array for ensemble
                        file_name = timestamp + "_" + str(batch_num)
                        np.save(file_name, prediction_flat)


                        #print(target_flat)
                        #print(np.mean(target_flat))
                        #print(np.max(target_flat))
                        #print(np.min(target_flat))

                        auc = roc_auc_score(target_flat, prediction_flat, sample_weight=mask_flat)
                        fprs, tprs, thresholds = roc_curve(target_flat, prediction_flat, sample_weight=mask_flat)
                        np_fprs, np_tprs, np_thresholds = np.array(fprs).flatten(), np.array(tprs).flatten(), np.array(thresholds).flatten()
                        fpr_10 = np_fprs[np.where(np_fprs < .10)]
                        tpr_10 = np_tprs[0:len(fpr_10)]

                        fpr_05 = np_fprs[np.where(np_fprs < .05)]
                        tpr_05 = np_tprs[0:len(fpr_05)]

                        fpr_025 = np_fprs[np.where(np_fprs < .025)]
                        tpr_025 = np_tprs[0:len(fpr_025)]

                        #upper_thresholds = np_thresholds[0:len(fpr_10)]
                        thresh_acc_strings = ""
                        list_fprs_tprs_thresholds = list(zip(fprs, tprs, thresholds))

                        auc_10_fpr = auc_(fpr_10, tpr_10)
                        auc_05_fpr = auc_(fpr_05, tpr_05)
                        auc_025_fpr = auc_(fpr_025, tpr_025)

                        thresh_max_items = "max thresh acc : {}, ".format(max_thresh_accuracy)

                        interval = 0.05
                        for i in np.arange(0, 1.0 + interval, interval):
                            index = int(round((len(thresholds) - 1) * i, 0))
                            fpr, tpr, threshold = list_fprs_tprs_thresholds[index]
                            thresh_acc = (1 - fpr) * test_neg_class_frac + tpr * test_pos_class_frac
                            thresh_acc_strings += "thresh: {}, thresh acc: {}, tpr: {}, spec: {}, ".format(threshold, thresh_acc, tpr, 1-fpr)

                        thresh_acc_strings = thresh_max_items +thresh_acc_strings
                        result_flat = (prediction_flat > decision_thresh).astype(int)

                        prediction_flat = np.round(prediction_flat)
                        target_flat = np.round(target_flat)


                        (precision, recall, fbeta_score, _) = precision_recall_fscore_support(target_flat,
                                                                                              prediction_flat,
                                                                                              average='binary',
                                                                                              sample_weight=mask_flat)

                        kappa = cohen_kappa_score(target_flat, prediction_flat, sample_weight=mask_flat)
                        tn, fp, fn, tp = confusion_matrix(target_flat, prediction_flat, sample_weight=mask_flat).ravel()

                        specificity = tn / (tn + fp)
                        #sess.run(tf.local_variables_initializer())
                        r_tn, r_fp, r_fn, r_tp = confusion_matrix(target_flat, result_flat, sample_weight=mask_flat).ravel()

                        r_acc = (r_tp+r_tn)/(r_tp+r_tn+r_fp+r_fn)
                        r_recall = r_tp / (r_tp + r_fn)
                        r_precision = r_tp / (r_tp + r_fp)
                        r_specificity = r_tn / (r_tn + r_fp)

                        # test_accuracy1 = test_accuracy1/len(test_inputs)
                        print(
                        'Step {}, test accuracy: {}, test accuracy: {}, thresh accuracy: {}, thresh recall: {}, thresh precision: {}, thresh specificity: {}, cost_unweighted: {} recall {}, specificity {}, precision {}, fbeta_score {}, auc {}, auc_10_fpr {}, auc_05_fpr {}, auc_025_fpr {}, kappa {}, class balance {}'.format(
                            batch_num, test_accuracy, r_acc, r_recall, r_precision, r_specificity, cost, cost_unweighted,recall, specificity, precision, fbeta_score, auc, auc_10_fpr, auc_05_fpr, auc_025_fpr, kappa, neg_pos_class_ratio))
                        # print('Step {}, test accuracy1: {}'.format(batch_num, test_accuracy1))

                        # n_examples = 5

                        # print(dataset.test_inputs.shape)
                        # print(len( dataset.test_inputs.tolist()))
                        # print(dataset.test_targets.shape)
                        # print(len(dataset.test_targets.tolist()))

                        t_inputs, t_masks, t_targets = dataset.test_inputs.tolist()[
                                                       :n_examples], dataset.test_masks.tolist()[
                                                                     :n_examples], dataset.test_targets.tolist()[
                                                                                   :n_examples]
                        test_segmentation = []
                        for i in range(n_examples):
                            test_i = np.multiply(t_inputs[i:(i + 1)], 1.0 / 255)
                            t_mask_i = t_masks[i:(i + 1)]
                            segmentation = sess.run(network.segmentation_result, feed_dict={
                                network.inputs: np.reshape(test_i, [1, Mod_WIDTH, Mod_HEIGHT, 1]),
                                network.masks: np.reshape(t_mask_i, [1, IMAGE_WIDTH, IMAGE_HEIGHT, 1])})
                            test_segmentation.append(segmentation[0])

                        test_plot_buf = draw_results(t_inputs[:n_examples],
                                                     np.multiply(t_targets[:n_examples], 1.0 / 255), test_segmentation,
                                                     test_accuracy, network, batch_num, decision_thresh)

                        image = tf.image.decode_png(test_plot_buf.getvalue(), channels=4)
                        image = tf.expand_dims(image, 0)
                        image_summary_op = tf.summary.image("plot", image)
                        image_summary = sess.run(image_summary_op)
                        summary_writer.add_summary(image_summary)
                        f1 = open(output_file, 'a')

                        test_accuracies.append((test_accuracy, batch_num))
                        test_auc.append((auc, batch_num))
                        test_auc_10_fpr.append((auc_10_fpr, batch_num))
                        test_auc_05_fpr.append((auc_05_fpr, batch_num))
                        test_auc_025_fpr.append((auc_025_fpr, batch_num))
                        max_thresh_accuracies.append((max_thresh_accuracy, batch_num))
                        print("Accuracies in time: ", [test_accuracies[x][0] for x in range(len(test_accuracies))])
                        print(test_accuracies)
                        max_acc = max(test_accuracies)
                        max_auc = max(test_auc)
                        max_auc_10_fpr = max(test_auc_10_fpr)
                        max_auc_05_fpr = max(test_auc_05_fpr)
                        max_auc_025_fpr = max(test_auc_025_fpr)
                        max_thresh_accuracy = max(max_thresh_accuracies)
                        if batch_num % layer_output_freq == 0:
                            print("layer 1 pos-neg distribution")
                            print("total positive sum: "+str(total_pos))
                            print("total negative sum: "+str(total_neg))
                            print("percentage positive: " + str(float(total_pos)/float(total_neg+total_pos)))
                            print("percentage negative: " + str(float(total_neg)/float(total_neg+total_pos)))
                            if len(channel_list) > 0:
                                tile_images(channel_list, file_name=cur_time+"_layer1_collage.jpeg")
                        print("Best accuracy: {} in batch {}".format(max_acc[0], max_acc[1]))
                        print("Total time: {}".format(time.time() - global_start))
                        f1.write(
                            'Step {}, test accuracy: {}, thresh accuracy: {}, thresh recall: {}, thresh precision: {}, thresh specificity: {}, cost: {}, cost_unweighted: {}, recall {}, specificity {}, auc {}, auc_10_fpr {}, auc_05_fpr {}, auc_025_fpr {}, precision {}, fbeta_score {}, kappa {}, class balance {}, '
                            'max acc {} {}, max auc {} {}, max auc 10 fpr {} {}, max auc 5 fpr {} {}, max auc 2.5 fpr {} {}, sample test image {} \n'.format(
                                batch_num, test_accuracy, r_acc, r_recall, r_precision, r_specificity, cost, cost_unweighted, recall, specificity, auc, auc_10_fpr, auc_05_fpr, auc_025_fpr, precision, fbeta_score, kappa, neg_pos_class_ratio,
                                max_acc[0],max_acc[1], max_auc[0], max_auc[1], max_auc_10_fpr[0], max_auc_10_fpr[1], max_auc_05_fpr[0], max_auc_05_fpr[0], max_auc_025_fpr[0], max_auc_025_fpr[1], sample_test_image))
                        f1.write(('Step {}, '+"overall max thresh accuracy {} {}, ".format(max_thresh_accuracy[0], max_thresh_accuracy[1])+thresh_acc_strings+'\n').format(batch_num))
                        f1.close()
                        f1 = open(cost_log, 'a')
                        f1.write('Step {}, training cost {}, training cost unweighted {}, test cost {}, test cost unweighted {}\n'.format(batch_num, cost, cost_unweighted, test_cost, test_cost_unweighted))
                        f1.close()

n_examples = 5
if __name__ == '__main__':
    ensemble_count = 1
    start_constant = 1.0
    if ensemble_count == 1:
        tuning_constants = [start_constant]
    else:
        end_constant = 1.5
        interval = (end_constant - start_constant) / float(ensemble_count - 1)
        tuning_constants = list(np.arange(start_constant, end_constant+interval, interval))
    for i in range(ensemble_count):
        tuning_constant = tuning_constants[i]
        cur_time = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")

        output_file = cur_time+"_results.txt"
        cost_log = cur_time+"_cost_log.txt"

        f1 = open(output_file, 'w')
        f1.close()
        f1 = open(cost_log, 'w')
        f1.close()
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
        kwargs = {'score_freq': 200, 'end_freq': 1000, 'layer_output_freq': 800, 'decision_thresh': .75, 'output_file': output_file, 'cost_log': cost_log, 'tuning_constant': tuning_constant, 'cur_time': cur_time}
        p = multiprocessing.Process(target=train, kwargs=kwargs)
        p.start()
        p.join()