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
import pydensecrf.densecrf as dcrf
from sklearn.model_selection import KFold, cross_val_score
import random
from pydensecrf.utils import compute_unary, create_pairwise_bilateral, create_pairwise_gaussian, softmax_to_unary

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

def dice_coe(output, target, loss_type='jaccard', axis=None, smooth=1e-5):
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
    ##
    dice = tf.reduce_mean(dice)
    return dice



def dice_hard_coe(output, target, threshold=0.5, axis=None, smooth=1e-5):
    output = tf.cast(output > threshold, dtype=tf.float32)
    target = tf.cast(target > threshold, dtype=tf.float32)
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
    hard_dice = tf.reduce_mean(hard_dice)
    return hard_dice



def iou_coe(output, target, threshold=0.5, axis=None, smooth=1e-5):

    pre = tf.cast(output > threshold, dtype=tf.float32)
    truth = tf.cast(target > threshold, dtype=tf.float32)
    inse = tf.reduce_sum(tf.multiply(pre, truth), axis=axis) # AND
    union = tf.reduce_sum(tf.cast(tf.add(pre, truth)>= 1, dtype=tf.float32), axis=axis) # OR
    ## old axis=[0,1,2,3]
    # epsilon = 1e-5
    # batch_iou = inse / (union + epsilon)
    ## new haodong
    batch_iou = (inse + smooth) / (union + smooth)
    iou = tf.reduce_mean(batch_iou)
    return iou#, pre, truth, inse, union



class Network:
    IMAGE_HEIGHT = 1024
    IMAGE_WIDTH = 1024
    IMAGE_CHANNELS = 1

    def __init__(self, net_id, layers = None, per_image_standardization=True, batch_norm=True, skip_connections=True):
        # Define network - ENCODER (decoder will be symmetric).

        if layers == None:
            layers = []
            layers.append(Conv2d(kernel_size=3, strides=[1, 1, 1, 1], output_channels=64, name='conv_1_1', net_id = net_id))
            #layers.append(Conv2d(kernel_size=3, strides=[1, 1, 1, 1], output_channels=64, name='conv_1_2', net_id = net_id))
            layers.append(MaxPool2d(kernel_size=2, name='max_1', skip_connection=True and skip_connections))

            layers.append(Conv2d(kernel_size=3, strides=[1, 1, 1, 1], output_channels=128, name='conv_2_1', net_id = net_id))
            #layers.append(Conv2d(kernel_size=3, strides=[1, 1, 1, 1], output_channels=128, name='conv_2_2'))

            layers.append(MaxPool2d(kernel_size=2, name='max_2', skip_connection=True and skip_connections))

            layers.append(Conv2d(kernel_size=3, strides=[1, 1, 1, 1], output_channels=256, name='conv_3_1', net_id = net_id))
            layers.append(Conv2d(kernel_size=3, strides=[1, 1, 1, 1], output_channels=256, name='conv_3_2', net_id = net_id))
            #layers.append(Conv2d(kernel_size=3, strides=[1, 1, 1, 1], output_channels=256, name='conv_3_3'))

            layers.append(MaxPool2d(kernel_size=2, name='max_3', skip_connection=True and skip_connections))

            layers.append(Conv2d(kernel_size=3, strides=[1, 1, 1, 1], output_channels=512, name='conv_4_1', net_id = net_id))
            layers.append(Conv2d(kernel_size=3, strides=[1, 1, 1, 1], output_channels=512, name='conv_4_2', net_id = net_id))
            #layers.append(Conv2d(kernel_size=3, strides=[1, 1, 1, 1], output_channels=512, name='conv_4_3'))

            layers.append(MaxPool2d(kernel_size=2, name='max_4', skip_connection=True and skip_connections))

            layers.append(Conv2d(kernel_size=3, strides=[1, 1, 1, 1], output_channels=512, name='conv_5_1', net_id = net_id))
            layers.append(Conv2d(kernel_size=3, strides=[1, 1, 1, 1], output_channels=512, name='conv_5_2', net_id = net_id))
            #layers.append(Conv2d(kernel_size=3, strides=[1, 1, 1, 1], output_channels=512, name='conv_5_3'))

            layers.append(MaxPool2d(kernel_size=2, name='max_5', skip_connection=True and skip_connections))

            layers.append(Conv2d(kernel_size=7, strides=[1, 1, 1, 1], output_channels=4096, name='conv_6_1', net_id = net_id))
            layers.append(Conv2d(kernel_size=1, strides=[1, 1, 1, 1], output_channels=4096, name='conv_6_2', net_id = net_id))
            #layers.append(Conv2d(kernel_size=1, strides=[1, 1, 1, 1], output_channels=1000, name='conv_6_3'))            

        self.inputs = tf.placeholder(tf.float32, [None, self.IMAGE_HEIGHT, self.IMAGE_WIDTH, self.IMAGE_CHANNELS],
                                     name='inputs')
        self.targets = tf.placeholder(tf.float32, [None, self.IMAGE_HEIGHT, self.IMAGE_WIDTH, 1], name='targets')
        self.is_training = tf.placeholder_with_default(False, [], name='is_training')
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
        #Conv2d.reverse_global_variables()

        # DECODER
        for layer in layers:
            net = layer.create_layer_reversed(net, prev_layer=self.layers[layer.name])

        self.segmentation_result = tf.sigmoid(net)

        # segmentation_as_classes = tf.reshape(self.y, [50 * self.IMAGE_HEIGHT * self.IMAGE_WIDTH, 1])
        # targets_as_classes = tf.reshape(self.targets, [50 * self.IMAGE_HEIGHT * self.IMAGE_WIDTH])
        # print(self.y.get_shape())
        # self.cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(segmentation_as_classes, targets_as_classes))
        print('segmentation_result.shape: {}, targets.shape: {}'.format(self.segmentation_result.get_shape(),
                                                                        self.targets.get_shape()))

        # MSE loss - change to log loss
        
        self.cost = tf.losses.log_loss(self.targets,self.segmentation_result)
        #= tf.sqrt(tf.reduce_mean(tf.square(self.segmentation_result - self.targets)))
        self.train_op = tf.train.AdamOptimizer().minimize(self.cost)
        with tf.name_scope('accuracy'):
            #seg_result = self.segmentation_result.eval()
            #print(seg_result)
            #print(seg_result.shape)
            #inputs = self.inputs.eval()
            #print(inputs)
            #print(inputs.shape)
            #result = post_process_crf(seg_result, inputs)
            argmax_probs = tf.round(self.segmentation_result)  # 0x1
            correct_pred = tf.cast(tf.equal(argmax_probs, self.targets), tf.float32)
            self.accuracy = tf.reduce_mean(correct_pred)

            tf.summary.scalar('accuracy', self.accuracy)

        self.summaries = tf.summary.merge_all()


class Dataset:
    def __init__(self, batch_size, folder='vessels', include_hair=True):
        self.folder = folder
        self.batch_size = batch_size
        self.include_hair = include_hair

        #train_files, validation_files, test_files = self.train_valid_test_split(os.listdir(os.path.join(folder, 'inputs')))

        self.train_inputs, self.train_targets = ([],[])
        self.test_inputs, self.test_targets = ([],[])

        self.pointer = 0

    def file_paths_to_images(self, folder, file_indices, files_list, verbose=False):
        inputs = []
        targets = []

        for file_index in file_indices:
            file = files_list[file_index]
            input_image = os.path.join(folder, 'inputs', file)
            target1_image = os.path.join(folder, 'targets1', file)
            target2_image = os.path.join(folder, 'targets2', file)

            test_image = cv2.imread(input_image, 0)
            inputs.append(test_image)

            #print(np.array(skio.imread(target_image)).shape)

            if os.path.exists(target1_image):

                # load grayscale
                # test_image = np.multiply(test_image, 1.0 / 255)

                target1_image = np.array(skio.imread(target1_image))
                target1_image = cv2.threshold(target1_image, 127, 1, cv2.THRESH_BINARY_INV)[1]
                
                targets.append(target1_image)

            elif os.path.exists(target2_image):
                
                target2_image = np.array(skio.imread(target2_image))[:,:,3]
                target2_image = cv2.threshold(target2_image, 127, 1, cv2.THRESH_BINARY_INV)[1]
                
                targets.append(target2_image)
            else:
                print(target1_image)
                print(target2_image)
                print("here")
                
        return inputs, targets

    def train_valid_test_split(self, X, ratio=None):
        if ratio is None:
            ratio = (0.5, .25, .25)

        N = len(X)
        return (
            X[:int(ceil(N * ratio[0]))],
            X[int(ceil(N * ratio[0])): int(ceil(N * ratio[0] + N * ratio[1]))],
            X[int(ceil(N * ratio[0] + N * ratio[1])):]
        )
    """
    def cross_valid_test_split(self, split_num = 4):
        k_fold = KFold(n_splits=split_num)
        return k_fold
    """
    def num_batches_in_epoch(self):
        return int(math.floor(len(self.train_inputs) / self.batch_size))

    def reset_batch_pointer(self):
        #permutation = np.random.permutation(len(self.train_inputs))
        #self.train_inputs = [self.train_inputs[i] for i in permutation]
        #self.train_targets = [self.train_targets[i] for i in permutation]

        self.pointer = 0

    def next_batch(self):
        inputs = []
        targets = []
        # print(self.batch_size, self.pointer, self.train_inputs.shape, self.train_targets.shape)
        for i in range(self.batch_size):
            inputs.append(np.array(self.train_inputs[self.pointer + i]))
            targets.append(np.array(self.train_targets[self.pointer + i]))

        self.pointer += self.batch_size

        return np.array(inputs, dtype=np.uint8), np.array(targets, dtype=np.uint8)

    @property
    def test_set(self):
        return np.array(self.test_inputs, dtype=np.uint8), np.array(self.test_targets, dtype=np.uint8)


def draw_results(test_inputs, test_targets, test_segmentation, test_accuracy, network, batch_num):
    n_examples_to_plot = 12
    fig, axs = plt.subplots(4, n_examples_to_plot, figsize=(n_examples_to_plot * 3, 10))
    fig.suptitle("Accuracy: {}, {}".format(test_accuracy, network.description), fontsize=20)
    for example_i in range(n_examples_to_plot):
        axs[0][example_i].imshow(test_inputs[example_i], cmap='gray')
        #print(np.sum(test_targets[example_i].astype(np.float32)))
        axs[1][example_i].imshow(test_targets[example_i].astype(np.float32), cmap='gray')
        axs[2][example_i].imshow(
            np.reshape(test_segmentation[example_i], [network.IMAGE_HEIGHT, network.IMAGE_WIDTH]),
            cmap='gray')
        
        test_image_thresholded = np.array(
            [0 if x < 0.5 else 255 for x in test_segmentation[example_i].flatten()])
        axs[3][example_i].imshow(
            np.reshape(test_image_thresholded, [network.IMAGE_HEIGHT, network.IMAGE_WIDTH]),
            cmap='gray')

        #cv2.imwrite('{}figure_input{}.jpg'.format(batch_num, example_i), test_inputs[example_i]*255)
        #cv2.imwrite('{}figure_target{}.jpg'.format(batch_num, example_i), test_targets[example_i]*255)
        #cv2.imwrite('{}figure_segmentation{}.jpg'.format(batch_num, example_i), 255*np.reshape(test_segmentation[example_i], [network.IMAGE_HEIGHT, network.IMAGE_WIDTH]))
        #cv2.imwrite('{}figure_threshold{}.jpg'.format(batch_num, example_i), np.reshape(test_image_thresholded, [network.IMAGE_HEIGHT, network.IMAGE_WIDTH]))

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)

    IMAGE_PLOT_DIR = 'image_plots/'
    if not os.path.exists(IMAGE_PLOT_DIR):
        os.makedirs(IMAGE_PLOT_DIR)

    plt.savefig('{}/figure{}.jpg'.format(IMAGE_PLOT_DIR, batch_num))
    return buf

def train(train_indices, validation_indices):
    BATCH_SIZE = 1
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")

    dataset = Dataset(folder='vessels', include_hair=True,
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
    count = 0

    with tf.device('/gpu:1'):
        #with tf.device('/cpu:0'):
        network = Network(count)
    count +=1
        
    # create directory for saving models
    os.makedirs(os.path.join('save', network.description+str(count), timestamp))

    train_inputs, train_targets = dataset.file_paths_to_images(folder, train_indices, os.listdir(os.path.join(folder, 'inputs')))
    test_inputs, test_targets = dataset.file_paths_to_images(folder, validation_indices, os.listdir(os.path.join(folder, 'inputs')), True)

    dataset.train_inputs = train_inputs
    dataset.train_targets = train_targets
    dataset.test_inputs = test_inputs
    dataset.test_targets = test_targets

    # test_inputs, test_targets = test_inputs[:100], test_targets[:100]
    test_inputs = np.reshape(test_inputs, (-1, 1024, 1024, 1))
    test_targets = np.reshape(test_targets, (-1, 1024, 1024, 1))
    test_inputs = np.multiply(test_inputs, 1.0 / 255)

    #config = tf.ConfigProto(device_count = {'GPU': 0,'GPU': 1})

    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        with tf.device('/gpu:0'):
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
                if batch_num > 100:
                    epoch_i = 0
                    dataset.reset_batch_pointer()
                    break
                dataset.reset_batch_pointer()
                for batch_i in range(dataset.num_batches_in_epoch()):
                    batch_num = epoch_i * dataset.num_batches_in_epoch() + batch_i + 1
                    if batch_num > 100:
                        break

                    augmentation_seq_deterministic = augmentation_seq.to_deterministic()

                    start = time.time()
                    batch_inputs, batch_targets = dataset.next_batch()
                    batch_inputs = np.reshape(batch_inputs, (dataset.batch_size, network.IMAGE_HEIGHT, network.IMAGE_WIDTH, 1))
                    batch_targets = np.reshape(batch_targets, (dataset.batch_size, network.IMAGE_HEIGHT, network.IMAGE_WIDTH, 1))

                    batch_inputs = augmentation_seq_deterministic.augment_images(batch_inputs)
                    batch_inputs = np.multiply(batch_inputs, 1.0 / 255)

                    batch_targets = augmentation_seq_deterministic.augment_images(batch_targets, hooks=hooks_binmasks)
                    #with tf.device('/gpu:0'):
                    cost, _ = sess.run([network.cost, network.train_op], feed_dict={network.inputs: batch_inputs, network.targets: batch_targets, network.is_training: True})
                    end = time.time()
                    print('{}/{}, epoch: {}, cost: {}, batch time: {}'.format(batch_num, n_epochs * dataset.num_batches_in_epoch(), epoch_i, cost, end - start))
                    if batch_num % 10 == 0 or batch_num == n_epochs * dataset.num_batches_in_epoch():
                        test_accuracy = 0.0
                        test_accuracy1 = 0.0

                        target_array = np.zeros((len(test_inputs), 1024, 1024))
                        prediction_array = np.zeros((len(test_inputs), 1024, 1024))
                        
                        for i in range(len(test_inputs)):
                            inputs, results, targets, _, acc = sess.run([network.inputs, network.segmentation_result, network.targets, network.summaries, network.accuracy], feed_dict={network.inputs: test_inputs[i:(i+1)], network.targets: test_targets[i:(i+1)], network.is_training: False})

                            results = results[0,:,:,0]
                            inputs = inputs[0,:,:,0]
                            targets = targets[0,:,:,0]

                            target_array[i]=targets
                            prediction_array[i]=results

                            new_results = np.zeros((2,1024,1024))
                            new_results[0] = results
                            new_results[1] = 1-results
                        
                            #crf_result = post_process_crf(inputs, new_results)

                            #argmax_probs = np.round(crf_result)  # 0x1
                            #correct_pred = np.sum(argmax_probs == targets)

                            #acc1 = correct_pred/(1024*1024)
                            test_accuracy += acc
                            #test_accuracy1 += acc1

                        prediction_array = tf.convert_to_tensor(prediction_array, dtype=tf.float32)
                        target_array = tf.convert_to_tensor(target_array, dtype=tf.float32)

                        """
                        sess.run(tf.local_variables_initializer())

                        dice_coe_val = tf.global_variables_initializer()
                        hard_dice_val = tf.global_variables_initializer()
                        iou_coe_val = tf.global_variables_initializer()
                        recall =  tf.global_variables_initializer()
                        precision = tf.global_variables_initializer()
                        auc = tf.global_variables_initializer()
                        TP = tf.global_variables_initializer()
                        FP = tf.global_variables_initializer()
                        FN = tf.global_variables_initializer()
                        TN = tf.global_variables_initializer()
                        specificity = tf.global_variables_initializer()
                        """
                        
                        dice_coe_val = dice_coe(prediction_array, target_array)
                        hard_dice_coe_val = dice_hard_coe(prediction_array, target_array)
                        iou_coe_val = iou_coe(prediction_array, target_array)
                        recall = tf.metrics.recall(target_array, prediction_array)[0]
                        precision = tf.metrics.precision(target_array, prediction_array)[0]
                        auc = tf.metrics.auc(target_array, prediction_array)[0]
                        TP = tf.metrics.true_positives(target_array, prediction_array)[0]
                        FP = tf.metrics.false_positives(target_array, prediction_array)[0]
                        FN = tf.metrics.false_negatives(target_array, prediction_array)[0]
                        TN = 1024*1024-TP-FP-FN
                        specificity = TN/(TN+FP)

                        sess.run(tf.local_variables_initializer())
                        #sess.run(tf.global_variables_initializer())
                        
                        #print(dice_coe_val)
                        test_accuracy = test_accuracy/len(test_inputs)
                        #test_accuracy1 = test_accuracy1/len(test_inputs)
                        print('Step {}, test accuracy: {}, dice_coe {}, hard_dice {}, iou_coe {}, recall {}, precision {}, auc {}, specificity {}'.format(batch_num, test_accuracy, dice_coe_val.eval(), hard_dice_coe_val.eval(), iou_coe_val.eval(), recall.eval(), precision.eval(), auc.eval(), specificity.eval()))
                        #print('Step {}, test accuracy1: {}'.format(batch_num, test_accuracy1))
                        n_examples = 12

                        t_inputs, t_targets = dataset.test_inputs[:n_examples], dataset.test_targets[:n_examples]
                        test_segmentation = []
                        for i in range(n_examples):
                            test_i = np.multiply(t_inputs[i:(i+1)], 1.0 / 255)
                            segmentation = sess.run(network.segmentation_result, feed_dict={network.inputs: np.reshape(test_i, [1, network.IMAGE_HEIGHT, network.IMAGE_WIDTH, 1])})
                            test_segmentation.append(segmentation[0])                            

                        test_plot_buf = draw_results(t_inputs[:n_examples], np.multiply(t_targets[:n_examples],1.0/255), test_segmentation, test_accuracy, network, batch_num)

                        image = tf.image.decode_png(test_plot_buf.getvalue(), channels=4)
                        image = tf.expand_dims(image, 0)
                        image_summary_op = tf.summary.image("plot", image)
                        image_summary = sess.run(image_summary_op)
                        summary_writer.add_summary(image_summary)
                        f1 = open('out1.txt','a')
                        f2 = open('out2.txt','a')

                        test_accuracies.append((test_accuracy, batch_num))
                        test_accuracies1.append((test_accuracy1, batch_num))
                        print("Accuracies in time: ", [test_accuracies[x][0] for x in range(len(test_accuracies))])
                        print(test_accuracies)
                        max_acc = max(test_accuracies)
                        print("Best accuracy: {} in batch {}".format(max_acc[0], max_acc[1]))
                        print("Total time: {}".format(time.time() - global_start))
                        #f1.write("batch num: " + str(batch_num) + " " +str(test_accuracy) + " max: " + str(max_acc[0]) +" "+str(max_acc[1])+ "\n")
                        f1.write('Step {}, test accuracy: {}, dice_coe {}, hard_dice {}, iou_coe {}, recall {}, precision {}, auc {}, specificity {}, max acc {} {}'.format(batch_num, test_accuracy, dice_coe_val.eval(), hard_dice_coe_val.eval(), iou_coe_val.eval(), recall.eval(), precision.eval(), auc.eval(), specificity.eval(), max_acc[0], max_acc[1]))

                        print("Accuracies1 in time: ", [test_accuracies1[x][0] for x in range(len(test_accuracies1))])
                        print(str(test_accuracies1))
                        max_acc = max(test_accuracies1)
                        print("Best accuracy1: {} in batch {}".format(max_acc[0], max_acc[1]))
                        print("Total time: {}".format(time.time() - global_start))
                        #f2.write("batch num: " + str(batch_num) + " " +str(test_accuracy1) + " max: " + str(max_acc[0]) +" "+str(max_acc[1]) +"\n")
                        f2.write('Step {}, test accuracy: {}, dice_coe {}, hard_dice {}, iou_coe {}, recall {}, precision {}, auc {}, specificity {}, max acc {} {}'.format(batch_num, test_accuracy1, dice_coe_val.eval(), hard_dice_coe_val.eval(), iou_coe_val.eval(), recall.eval(), precision.eval(), auc.eval(), specificity.eval(), max_acc[0], max_acc[1]))
                        f1.close() 
                        f2.close()
                        break

def post_process_crf(input_it, prediction_it):
    #for input_t, prediction_it in zip(inputs, predictions):
    #also set kernel weights
    unary = softmax_to_unary(prediction_it)
    unary = np.ascontiguousarray(unary)
    d = dcrf.DenseCRF(1024*1024, 2)
    d.setUnaryEnergy(unary)
    feats = create_pairwise_gaussian(sdims=(3, 3), shape=(1024,1024))
    d.addPairwiseEnergy(feats, compat=3, kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)
    feats = create_pairwise_bilateral(sdims=(10, 10), schan=(.01), img=input_it, chdim=-1)
    d.addPairwiseEnergy(feats, compat=10, kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)
    Q = d.inference(10)
    res = np.argmax(Q, axis=0).reshape((1024, 1024))
    return (1-res)
    
if __name__ == '__main__':
    x = random.randint(1,100)                                     
    k_fold = KFold(n_splits=3, shuffle=True, random_state=x)

    f1 = open('out1.txt','w')
    f2 = open('out2.txt','w')

    f1.close() 
    f2.close()
    count = 0
    for train_indices, validation_indices in k_fold.split(os.listdir(os.path.join('vessels', 'inputs'))):
        p = multiprocessing.Process(target=train, args=(train_indices, validation_indices))
        p.start()
        p.join()
        count+=1
        if count > 0:
            break
