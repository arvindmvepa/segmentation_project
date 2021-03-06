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

from imgaug import augmenters as iaa
from imgaug import imgaug
from libs.activations import lrelu
from libs.utils import corrupt
from conv2d import Conv2d
from max_pool_2d import MaxPool2d
import datetime
import io
from PIL import Image
from skimage import io as skio
from sklearn.model_selection import KFold, cross_val_score
import random
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from PIL import Image

IMAGE_HEIGHT = 565
#IMAGE_HEIGHT = 584
IMAGE_WIDTH = 584
#INPUT_IMAGE_HEIGHT = 600
#INPUT_IMAGE_WIDTH = 600
INPUT_IMAGE_HEIGHT = IMAGE_HEIGHT
INPUT_IMAGE_WIDTH = IMAGE_WIDTH

Mod_HEIGHT = 584
Mod_WIDTH = 584

n_examples = 5

#np.set_printoptions(threshold=np.nan)

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
def mask_op_and_mask_mean(correct_pred, mask, num_batches = 1, width = IMAGE_WIDTH, height = IMAGE_HEIGHT):
    correct_pred = tf.multiply(correct_pred, mask)
    return mask_mean(tf.count_nonzero(correct_pred, dtype=tf.float32), mask, num_batches, width, height)

def mask_op_and_mask_mean_diff(correct_pred, mask, num_batches = 1, width = IMAGE_WIDTH, height = IMAGE_HEIGHT):
    correct_pred = tf.multiply(correct_pred, mask)
    return mask_mean(tf.count_nonzero(correct_pred, dtype=tf.float32), mask, num_batches, width, height)

def mask_mean(masked_pred, mask, num_batches = 1, width = IMAGE_WIDTH, height = IMAGE_HEIGHT):
    ones = tf.ones([num_batches, width, height, 1], tf.float32)
    FOV_num_pixels = tf.count_nonzero(tf.cast(tf.equal(mask, ones), tf.float32), dtype = tf.float32)
    return tf.divide(masked_pred, FOV_num_pixels)

def mask_mean_diff(masked_pred, mask, num_batches = 1, width = IMAGE_WIDTH, height = IMAGE_HEIGHT):
    ones = tf.ones([num_batches, width, height], tf.float32)
    FOV_num_pixels = tf.count_nonzero(tf.cast(tf.equal(mask, ones), tf.float32), dtype = tf.float32)
    return tf.divide(masked_pred, FOV_num_pixels)

def find_positive_weight(targets, masks):
    total_pos = 0
    total_num_pixels = 0
    total_neg = 0
    for target,mask in zip(targets,masks):
        target = np.multiply(target,mask)
        total_pos += np.count_nonzero(target)
        total_num_pixels += np.count_nonzero(mask)
    total_neg = total_num_pixels - total_pos
    weight = total_neg/total_pos
    return weight    

def dice_coe(output, target, mask = None, num_batches = 1, loss_type='jaccard', axis=None, smooth=1e-5):
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



def dice_hard_coe(output, target, mask = None, num_batches = 1, threshold=0.5, axis=None, smooth=1e-5):
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



def iou_coe(output, target, mask = None, num_batches = 1, threshold=0.5, axis=None, smooth=1e-5):

    pre = tf.cast(output > threshold, dtype=tf.float32)
    truth = tf.cast(target > threshold, dtype=tf.float32)
    if mask != None:
        output = tf.multiply(output, mask)
        target = tf.multiply(target, mask)
    inse = tf.reduce_sum(tf.multiply(pre, truth), axis=axis) # AND
    union = tf.reduce_sum(tf.cast(tf.add(pre, truth)>= 1, dtype=tf.float32), axis=axis) # OR
    ## old axis=[0,1,2,3]
    # epsilon = 1e-5
    # batch_iou = inse / (union + epsilon)
    ## new haodong
    batch_iou = (inse + smooth) / (union + smooth)
    if mask != None:
        iou = mask_mean_diff(batch_iou, mask, num_batches)
    else:
        iou = tf.reduce_mean(batch_iou)
    return iou#, pre, truth, inse, union



class Network:
    #IMAGE_HEIGHT = 565
    IMAGE_HEIGHT = IMAGE_HEIGHT
    IMAGE_WIDTH = IMAGE_WIDTH
    #INPUT_IMAGE_HEIGHT = 600
    #INPUT_IMAGE_WIDTH = 600

    IMAGE_CHANNELS = 1

    def __init__(self, net_id, weight=1, layers = None, per_image_standardization=True, batch_norm=True, skip_connections=True):
        # Define network - ENCODER (decoder will be symmetric).

        if layers == None:
            layers = []
            layers.append(Conv2d(kernel_size=3, output_channels=64, name='conv_1_1', net_id = net_id))
            #layers.append(Conv2d(kernel_size=3, strides=[1, 1, 1, 1], output_channels=64, name='conv_1_2', net_id = net_id))
            layers.append(MaxPool2d(kernel_size=2, name='max_1', skip_connection=True and skip_connections))

            layers.append(Conv2d(kernel_size=3, output_channels=128, name='conv_2_1', net_id = net_id))
            #layers.append(Conv2d(kernel_size=3, strides=[1, 1, 1, 1], output_channels=128, name='conv_2_2'))

            layers.append(MaxPool2d(kernel_size=2, name='max_2', skip_connection=True and skip_connections))
            layers.append(Conv2d(kernel_size=3, output_channels=256, name='conv_3_1', net_id = net_id))
            layers.append(Conv2d(kernel_size=3, dilation = 2,  output_channels=256, name='conv_3_2', net_id = net_id))

            #layers.append(Conv2d(kernel_size=3, strides=[1, 1, 1, 1], output_channels=256, name='conv_3_3'))

            layers.append(MaxPool2d(kernel_size=2, name='max_3', skip_connection=True and skip_connections))

            #layers.append(Conv2d(kernel_size=3,  output_channels=512, name='conv_4_1', net_id = net_id))
            #layers.append(Conv2d(kernel_size=3, output_channels=512, name='conv_4_2', net_id = net_id))

            #layers.append(Conv2d(kernel_size=3, strides=[1, 1, 1, 1], output_channels=512, name='conv_4_3'))

            #layers.append(MaxPool2d(kernel_size=2, name='max_4', skip_connection=True and skip_connections))

            #layers.append(Conv2d(kernel_size=3, output_channels=512, name='conv_5_1', net_id = net_id))
            #layers.append(Conv2d(kernel_size=3, output_channels=512, name='conv_5_2', net_id = net_id))
            #layers.append(Conv2d(kernel_size=3, strides=[1, 1, 1, 1], output_channels=512, name='conv_5_3'))

            #layers.append(MaxPool2d(kernel_size=2, name='max_5', skip_connection=True and skip_connections))

            layers.append(Conv2d(kernel_size=7, output_channels=4096, name='conv_6_1', net_id = net_id))
            layers.append(Conv2d(kernel_size=1, output_channels=4096, name='conv_6_2', net_id = net_id))
            #layers.append(Conv2d(kernel_size=1, strides=[1, 1, 1, 1], output_channels=1000, name='conv_6_3'))
            #self.inputs = tf.placeholder(tf.float32, [None, self.IMAGE_WIDTH, self.IMAGE_HEIGHT, self.IMAGE_CHANNELS],name='inputs')
            self.inputs = tf.placeholder(tf.float32, [None, Mod_WIDTH, Mod_HEIGHT, self.IMAGE_CHANNELS],name='inputs')

        self.masks = tf.placeholder(tf.float32, [None, self.IMAGE_WIDTH, self.IMAGE_HEIGHT, 1], name='masks')
        self.targets = tf.placeholder(tf.float32, [None, self.IMAGE_WIDTH, self.IMAGE_HEIGHT, 1], name='targets')
        self.is_training = tf.placeholder_with_default(False, [], name='is_training')

        #has to change for multiple batches
        #self.ones = tf.ones([1, self.IMAGE_WIDTH, self.IMAGE_HEIGHT], tf.int32)

        self.description = ""

        self.layers = {}

        if per_image_standardization:
            list_of_images_norm = tf.map_fn(tf.image.per_image_standardization, self.inputs)
            net = tf.stack(list_of_images_norm)
        else:
            net = self.inputs

        # ENCODER
        for layer in layers:
            self.layers[layer.name] = net = layer.create_layer(net)
            self.description += "{}".format(layer.get_description())

        print("Current input shape: ", net.get_shape())

        layers.reverse()

        # DECODER
        last_layer = layers[len(layers)-1]
        for layer in layers:
            net = layer.create_layer_reversed(net, prev_layer=self.layers[layer.name])
        net = tf.image.resize_image_with_crop_or_pad(net,IMAGE_WIDTH,IMAGE_HEIGHT)
        net = tf.multiply(net, self.masks)
        self.segmentation_result = tf.sigmoid(net)

        self.targets = tf.multiply(self.targets, self.masks)

        print('segmentation_result.shape: {}, targets.shape: {}'.format(self.segmentation_result.get_shape(),
                                                                        self.targets.get_shape()))

        
        self.cost = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(self.targets, net, pos_weight=weight))
        print('net.shape: {}'.format(net.get_shape()))
        #= tf.sqrt(tf.reduce_mean(tf.square(self.segmentation_result - self.targets)))

        #override the methods called by minimize to debug the error
        #debug first layer to figure out what's going on

        self.train_op = tf.train.AdamOptimizer().minimize(self.cost)
        with tf.name_scope('accuracy'):
            #t_shape_list = (self.segmentation_result).get_shape().as_list()
            #num_batches = t_shape_list[0]
            num_batches = 1
            argmax_probs = tf.round(self.segmentation_result)  # 0x1
            correct_pred = tf.cast(tf.equal(argmax_probs, self.targets), tf.float32)
            #correct_pred = tf.multiply(correct_pred, self.masks)
            #FOV_num_pixels = tf.cast(tf.equal(self.masks, self.ones), tf.float32)
            #self.accuracy = tf.divide(correct_pred, FOV_num_pixels)
            self.accuracy = mask_op_and_mask_mean(correct_pred, self.masks, num_batches, IMAGE_WIDTH,IMAGE_HEIGHT)
            tf.summary.scalar('accuracy', self.accuracy)

        self.summaries = tf.summary.merge_all()

class Dataset:
    def __init__(self, batch_size, folder='drive', include_hair=True):
        self.folder = folder
        self.batch_size = batch_size
        self.include_hair = include_hair

        #train_files, validation_files, test_files = self.train_valid_test_split(os.listdir(os.path.join(folder, 'inputs')))

        self.train_inputs, self.train_masks, self.train_targets = ([],[],[])
        self.test_inputs, self.train_masks, self.test_targets = ([],[], [])

        self.pointer = 0

    def file_paths_to_images(self, folder, file_indices, files_list, verbose=False):
        inputs = []
        masks = []
        targets = []

        for file_index in file_indices:
            orig_file = files_list[file_index]

            num = int(orig_file[0:2])
            file = str(num) + "_manual1.gif"
            mask_file = str(num) +"_training_mask.gif"

            input_image = os.path.join(folder, 'inputs', orig_file)
            target1_image = os.path.join(folder, 'targets1', file)
            target2_image = os.path.join(folder, 'targets2', file)

            mask_loc = os.path.join(folder, 'masks', mask_file)

            #add training image to dataset
            test_image = cv2.imread(input_image, 1)
            test_image = test_image[:,:,1]
            top_pad = int((Mod_HEIGHT-IMAGE_HEIGHT)/2)
            bot_pad = (Mod_HEIGHT-IMAGE_HEIGHT) - top_pad
            left_pad = int((Mod_WIDTH-IMAGE_WIDTH)/2)
            right_pad = (Mod_WIDTH-IMAGE_WIDTH) - left_pad
            print("before:")
            print(test_image.shape)
            test_image = cv2.copyMakeBorder(test_image, left_pad, right_pad, top_pad, bot_pad, cv2.BORDER_CONSTANT, 0)
            print("after:")
            print(test_image.shape)
            inputs.append(test_image)

            #add mask for training image to dataset
            mask = Image.open(mask_loc)
            mask_array = np.array(mask)
            mask_array = mask_array/255
            masks.append(mask_array)

            if os.path.exists(target1_image):

                # load grayscale
                # test_image = np.multiply(test_image, 1.0 / 255)
                print(target1_image)

                target1_image = np.array(skio.imread(target1_image))
                target1_image = cv2.threshold(target1_image, 127, 1, cv2.THRESH_BINARY)[1]
                #target1_image = cv2.resize(target1_image, (IMAGE_WIDTH,IMAGE_HEIGHT))
                
                print(target1_image)
                
                targets.append(target1_image)

            elif os.path.exists(target2_image):
                
                target2_image = np.array(skio.imread(target2_image))[:,:,3]
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
        #permutation = np.random.permutation(len(self.train_inputs))
        #self.train_inputs = [self.train_inputs[i] for i in permutation]
        #self.train_targets = [self.train_targets[i] for i in permutation]

        self.pointer = 0

    def next_batch(self):
        inputs = []
        masks = []
        targets = []
        # print(self.batch_size, self.pointer, self.train_inputs.shape, self.train_targets.shape)
        for i in range(self.batch_size):
            inputs.append(np.array(self.train_inputs[self.pointer + i]))
            masks.append(np.array(self.train_masks[self.pointer + i]))
            targets.append(np.array(self.train_targets[self.pointer + i]))

        self.pointer += self.batch_size

        return np.array(inputs, dtype=np.uint8), np.array(masks, dtype=np.uint8), np.array(targets, dtype=np.uint8)

    @property
    def test_set(self):
        return np.array(self.test_inputs, dtype=np.uint8), np.array(self.test_targets, dtype=np.uint8)


def draw_results(test_inputs, test_targets, test_segmentation, test_accuracy, network, batch_num):
    n_examples_to_plot = n_examples
    fig, axs = plt.subplots(4, n_examples_to_plot, figsize=(n_examples_to_plot * 3, 10))
    fig.suptitle("Accuracy: {}, {}".format(test_accuracy, network.description), fontsize=20)
    for example_i in range(n_examples_to_plot):
        axs[0][example_i].imshow(test_inputs[example_i], cmap='gray')
        #print(np.sum(test_targets[example_i].astype(np.float32)))
        axs[1][example_i].imshow(test_targets[example_i].astype(np.float32), cmap='gray')
        axs[2][example_i].imshow(
            np.reshape(test_segmentation[example_i], [network.IMAGE_WIDTH, network.IMAGE_HEIGHT]),
            cmap='gray')
        
        test_image_thresholded = np.array(
            [0 if x < 0.5 else 255 for x in test_segmentation[example_i].flatten()])
        axs[3][example_i].imshow(
            np.reshape(test_image_thresholded, [network.IMAGE_WIDTH, network.IMAGE_HEIGHT]),
            cmap='gray')

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)

    IMAGE_PLOT_DIR = 'image_plots/'
    if not os.path.exists(IMAGE_PLOT_DIR):
        os.makedirs(IMAGE_PLOT_DIR)

    plt.savefig('{}/figure{}.jpg'.format(IMAGE_PLOT_DIR, batch_num))
    return buf

def train(train_indices, validation_indices, run_id):
    BATCH_SIZE = 1
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")

    dataset = Dataset(folder='drive', include_hair=True,
                      batch_size=BATCH_SIZE)

    #inputs, targets = dataset.next_batch()
    #print(inputs.shape, targets.shape)

    # augmentation_seq = iaa.Sequential([
    #     iaa.Crop(px=(0, 16)),  # crop images from each side by 0 to 16px (randomly chosen)
    #     iaa.Fliplr(0.5),  # horizontally flip 50% of the images
    #     iaa.GaussianBlur(sigma=(0, 2.0))  # blur images with a sigma of 0 to 3.0
    # ])

    augmentation_seq = iaa.Sequential([
        iaa.Crop(px=(0, 16), name="Cropper"),  # crop images from each side by 0 to 16px (randomly chosen)
        iaa.Fliplr(0.5, name="Flipper"),
        iaa.GaussianBlur((0, 3.0), name="GaussianBlur"),
        iaa.Dropout(0.02, name="Dropout"),
        iaa.AdditiveGaussianNoise(scale=0.01 * 255, name="GaussianNoise"),
        iaa.Affine(translate_px={"x": (-1024 // 3, 1024 // 3)}, name="Affine")
    ])

    # change the activated augmenters for binary masks,
    # we only want to execute horizontal crop, flip and affine transformation
    def activator_binmasks(images, augmenter, parents, default):
        if augmenter.name in ["GaussianBlur", "Dropout", "GaussianNoise"]:
            return False
        else:
            # default value for all other augmenters
            return default

    hooks_binmasks = imgaug.HooksImages(activator=activator_binmasks)

    k_fold = KFold(n_splits=4)
    folder = dataset.folder

    train_inputs, train_masks, train_targets = dataset.file_paths_to_images(folder, train_indices, os.listdir(os.path.join(folder, 'inputs')))
    test_inputs, test_masks, test_targets = dataset.file_paths_to_images(folder, validation_indices, os.listdir(os.path.join(folder, 'inputs')), True)
    ##DEBUG
    pos_weight = find_positive_weight(train_targets, train_masks)
    #pos_weight = 1

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
    #test_inputs = np.reshape(test_inputs, (len(test_inputs), Mod_HEIGHT, Mod_WIDTH, 1))
    test_inputs = np.reshape(test_inputs, (len(test_inputs), Mod_WIDTH, Mod_HEIGHT, 1))
    test_inputs = np.multiply(test_inputs, 1.0 / 255)
    test_targets = np.reshape(test_targets, (len(test_targets), IMAGE_WIDTH, IMAGE_HEIGHT, 1))
    test_masks = np.reshape(test_masks, (len(test_masks), IMAGE_WIDTH, IMAGE_HEIGHT, 1))

    #test_inputs = np.pad(test_inputs, ((8,8),(18,17)), 'constant', constant_values=0)
    #test_inputs=tf.image.resize_image_with_crop_or_pad(test_inputs,INPUT_IMAGE_HEIGHT,INPUT_IMAGE_WIDTH)
    #test_inputs=tf.image.resize_image_with_crop_or_pad(test_inputs,INPUT_IMAGE_HEIGHT,INPUT_IMAGE_WIDTH)
    

    #config = tf.ConfigProto(device_count = {'GPU': 0,'GPU': 1})

    count = 0
    with tf.device('/gpu:1'):
        #with tf.device('/cpu:0'):
        network = Network(net_id = count, weight=pos_weight)
    count +=1

    # create directory for saving model
    os.makedirs(os.path.join('save', network.description+str(count), timestamp))

    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
    config.gpu_options.allow_growth = True

    saver = tf.train.Saver()

    with tf.Session(config=config) as sess:
        with tf.device('/gpu:1'):
            print(sess.run(tf.global_variables_initializer()))
            
            summary_writer = tf.summary.FileWriter('{}/{}-{}'.format('logs', network.description, timestamp), graph=tf.get_default_graph())
            saver = tf.train.Saver(tf.all_variables(), max_to_keep=None)

            test_accuracies = []
            test_accuracies1 = []
            # Fit all training data
            n_epochs = 5000
            global_start = time.time()
            acc = 0.0
            batch_num = 0
            for epoch_i in range(n_epochs):
                if batch_num > 20000:
                    epoch_i = 0
                    dataset.reset_batch_pointer()
                    break
                dataset.reset_batch_pointer()
                for batch_i in range(dataset.num_batches_in_epoch()):
                    batch_num = epoch_i * dataset.num_batches_in_epoch() + batch_i + 1
                    if batch_num > 20000:
                        break

                    augmentation_seq_deterministic = augmentation_seq.to_deterministic()

                    start = time.time()
                    batch_inputs, batch_masks, batch_targets = dataset.next_batch()

                    batch_inputs = np.reshape(batch_inputs, (dataset.batch_size, Mod_WIDTH, Mod_HEIGHT, 1))
                    batch_masks = np.reshape(batch_masks, (dataset.batch_size, network.IMAGE_WIDTH, network.IMAGE_HEIGHT, 1))
                    batch_targets = np.reshape(batch_targets, (dataset.batch_size, network.IMAGE_WIDTH, network.IMAGE_HEIGHT, 1))


                    batch_inputs = augmentation_seq_deterministic.augment_images(batch_inputs)
                    batch_inputs = np.multiply(batch_inputs, 1.0 / 255)

                    batch_targets = augmentation_seq_deterministic.augment_images(batch_targets, hooks=hooks_binmasks)
                    cost, _ = sess.run([network.cost, network.train_op], feed_dict={network.inputs: batch_inputs, network.masks: batch_masks, network.targets: batch_targets, network.is_training: True})
                    end = time.time()

                    print('{}/{}, epoch: {}, cost: {}, batch time: {}, positive_weight: {}'.format(batch_num, n_epochs * dataset.num_batches_in_epoch(), epoch_i, cost, end - start, pos_weight))
                    if batch_num % 500 == 0 or batch_num == n_epochs * dataset.num_batches_in_epoch():
                        test_accuracy = 0.0

                        mask_array = np.zeros((len(test_inputs), IMAGE_WIDTH, IMAGE_HEIGHT))
                        target_array = np.zeros((len(test_inputs), IMAGE_WIDTH, IMAGE_HEIGHT))
                        prediction_array = np.zeros((len(test_inputs), IMAGE_WIDTH, IMAGE_HEIGHT))
                        
                        for i in range(len(test_inputs)):
                            #print(test_inputs.shape)
                            #print(test_targets.shape)
                            #print(test_masks.shape)
                            #print(test_inputs[i:(i+1)].shape)
                            #print(test_targets[i:(i+1)].shape)
                            #print(test_masks[i:(i+1)].shape)
                            #inputs, masks, results, targets, _, acc = sess.run([network.inputs, network.masks, network.segmentation_result, network.targets, network.summaries, network.accuracy],
                                                                               #feed_dict={network.inputs: test_inputs[i:(i+1)], network.masks: test_masks[i:(i+1)], network.targets: test_targets[i:(i+1)], network.is_training: False})
                            #inputs, masks, results, targets, _, acc = sess.run([network.inputs, network.masks, network.segmentation_result, network.targets, network.summaries, network.accuracy],
                                                                               #feed_dict={network.inputs: test_inputs[i:(i+1)], network.targets: test_targets[i:(i+1)], network.is_training: False})
                            inputs, masks, results, targets, acc = sess.run([network.inputs, network.masks, network.segmentation_result, network.targets, network.accuracy],
                                                                               feed_dict={network.inputs: test_inputs[i:(i+1)], network.masks: test_masks[i:(i+1)], network.targets: test_targets[i:(i+1)], network.is_training: False})

                            inputs = inputs[0,:,:,0]
                            masks = masks[0, :, :, 0]
                            results = results[0,:,:,0]
                            targets = targets[0,:,:,0]
                            mask_array[i] = masks
                            target_array[i]=targets
                            prediction_array[i]=results
                            test_accuracy += acc

                        test_accuracy = test_accuracy / len(test_inputs)

                        mask_tensor = tf.convert_to_tensor(mask_array, dtype=tf.float32)
                        prediction_tensor = tf.convert_to_tensor(prediction_array, dtype=tf.float32)
                        target_tensor = tf.convert_to_tensor(target_array, dtype=tf.float32)

                        dice_coe_val = dice_coe(prediction_tensor, target_tensor, mask_tensor, len(test_inputs))
                        hard_dice_coe_val = dice_hard_coe(prediction_tensor, target_tensor, mask_tensor, len(test_inputs))
                        iou_coe_val = iou_coe(prediction_tensor, target_tensor, mask_tensor, len(test_inputs))

                        mask_flat = mask_array.flatten()
                        prediction_flat = prediction_array.flatten()
                        target_flat = target_array.flatten()
                        auc = roc_auc_score(target_flat, prediction_flat, sample_weight = mask_flat)

                        prediction_flat = np.round(prediction_flat)
                        target_flat = np.round(target_flat)
                        (precision, recall, fbeta_score, _) = precision_recall_fscore_support(target_flat, prediction_flat, average='binary', sample_weight = mask_flat)

                        tn, fp, fn, tp = confusion_matrix(target_flat, prediction_flat).ravel()
                        specificity = tn / (tn+fp)
                        sess.run(tf.local_variables_initializer())

                        #test_accuracy1 = test_accuracy1/len(test_inputs)
                        print('Step {}, test accuracy: {}, dice_coe {}, hard_dice {}, iou_coe {}, recall {}, precision {}, fbeta_score {}, auc {}, specificity {}'.format(batch_num, test_accuracy, dice_coe_val.eval(), hard_dice_coe_val.eval(), iou_coe_val.eval(), recall, precision, fbeta_score, auc, specificity))
                        #print('Step {}, test accuracy1: {}'.format(batch_num, test_accuracy1))

                        #n_examples = 5

                        #print(dataset.test_inputs.shape)
                        #print(len( dataset.test_inputs.tolist()))
                        #print(dataset.test_targets.shape)
                        #print(len(dataset.test_targets.tolist()))
                        
                        t_inputs, t_masks, t_targets = dataset.test_inputs.tolist()[:n_examples], dataset.test_masks.tolist()[:n_examples], dataset.test_targets.tolist()[:n_examples]
                        test_segmentation = []
                        for i in range(n_examples):
                            test_i = np.multiply(t_inputs[i:(i+1)], 1.0 / 255)
                            t_mask_i = t_masks[i:(i+1)]
                            segmentation = sess.run(network.segmentation_result, feed_dict={network.inputs: np.reshape(test_i, [1, Mod_WIDTH, Mod_HEIGHT, 1]),
                                                                                            network.masks: np.reshape(t_mask_i, [1, IMAGE_WIDTH, IMAGE_HEIGHT, 1])})
                            test_segmentation.append(segmentation[0])                            

                        test_plot_buf = draw_results(t_inputs[:n_examples], np.multiply(t_targets[:n_examples],1.0/255), test_segmentation, test_accuracy, network, batch_num)

                        image = tf.image.decode_png(test_plot_buf.getvalue(), channels=4)
                        image = tf.expand_dims(image, 0)
                        image_summary_op = tf.summary.image("plot", image)
                        image_summary = sess.run(image_summary_op)
                        summary_writer.add_summary(image_summary)
                        f1 = open('out1.txt','a')

                        test_accuracies.append((test_accuracy, batch_num))
                        print("Accuracies in time: ", [test_accuracies[x][0] for x in range(len(test_accuracies))])
                        print(test_accuracies)
                        max_acc = max(test_accuracies)
                        print("Best accuracy: {} in batch {}".format(max_acc[0], max_acc[1]))
                        print("Total time: {}".format(time.time() - global_start))
                        f1.write('Step {}, test accuracy: {}, dice_coe {}, hard_dice {}, iou_coe {}, recall {}, precision {}, fbeta_score {}, auc {}, specificity {}, max acc {} {} \n'.format(batch_num, test_accuracy, dice_coe_val.eval(), hard_dice_coe_val.eval(), iou_coe_val.eval(), recall, precision, fbeta_score, auc, specificity, max_acc[0], max_acc[1]))
                        f1.close()

if __name__ == '__main__':
    x = random.randint(1,100)                                     
    k_fold = KFold(n_splits=3, shuffle=True, random_state=x)

    f1 = open('out1.txt','w')
    f2 = open('out2.txt','w')

    f1.close() 
    f2.close()

    count = 0
    for train_indices, validation_indices in k_fold.split(os.listdir(os.path.join('drive', 'inputs'))):
        f1 = open('out1.txt','a')
        f2 = open('out2.txt','a')
        f1.write('Train Indices {} Validation Indices {} \n'.format(train_indices, validation_indices))
        f2.write('Train Indices {} Validation Indices {} \n'.format(train_indices, validation_indices))
        f1.close()
        f2.close()
        p = multiprocessing.Process(target=train, args=(train_indices, validation_indices, count))
        p.start()
        p.join()
        count += 1
