print(__doc__)

import numpy as np
#98% threshhold

import math

from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler

import copy
from skimage import io

from PIL import Image
import numpy
from queue import PriorityQueue
import os
import glob

from shutil import copyfile
import random
from sklearn.model_selection import KFold, cross_val_score

#numpy.set_printoptions(threshold=numpy.nan)


#min distance is incorrect - fix later!!!!!!
#also decision on replacement color, white or the mean in the image

def test_clustering():
    x = random.randint(1,100)                                     
    k_fold = KFold(n_splits=3, shuffle=True, random_state=x)
    count = 0
    for train_indices, validation_indices in k_fold.split(os.listdir(os.path.join('vessels', 'inputs'))):
        results = two_stage_kmeans(train_indices)
        new_results = modify_validation_files(validation_indices, results[0])
        count += 1
        if count > 0:
            break

def modify_validation_files(file_indices, min_dist):
    path = os.path.join('vessels', 'val_transformed')
    files = glob.glob(path+"/*")

    for f in files:
        os.remove(f) 
    files_list = os.listdir(os.path.join('vessels', 'inputs'))

    noise_removed = 0
    segmented_noise_overall = 0
    net_noise_removed = 0
    segmentation_total = 0
    total_modified_images = 0
    
    for file_index in file_indices:
        
        file = files_list[file_index]
        new_path = os.path.join('vessels', 'val_transformed', file)

        input_image = os.path.join('vessels', 'inputs', file)
        copyfile(input_image, new_path)

        file_loc = input_image

        target1_image = os.path.join('vessels', 'targets1', file)
        target2_image = os.path.join('vessels', 'targets2', file)

        im = io.imread(file_loc)
        imarray = numpy.array(im)
        im_features = np.zeros((imarray.shape[0]*imarray.shape[1],1))
        i=0
        for x in range(imarray.shape[0]):
            for y in range(imarray.shape[1]):
                im_features[i,0]=imarray[x,y]
                i+=1

        n_clusters_=2

        db = KMeans(n_clusters=n_clusters_, random_state=0).fit(im_features)
        labels = db.labels_
        centers = db.cluster_centers_

        dark_cluster_label = np.argmin(centers)
        replacement_color = np.max(centers)

        new_image = np.zeros((imarray.shape[0], imarray.shape[1]), dtype=np.int8)

        scale = 255/3

        dark_cluster = []
        i = 0
        for x in range(imarray.shape[0]):
            for y in range(imarray.shape[1]):
                cur_lab = labels[i]
                if cur_lab == dark_cluster_label:
                    dark_cluster.append([i,x,y,imarray[x,y]])
                i+=1

        length_dark_cluster = len(dark_cluster)
        dark_cluster_features = np.zeros((length_dark_cluster,3))

        a = 1
        b = 1
        c = 2

        for i in range(length_dark_cluster):
            dark_cluster_features[i,0]= dark_cluster[i][1]*a
            dark_cluster_features[i,1]= dark_cluster[i][2]*b
            dark_cluster_features[i,2]= math.pow(math.pow((dark_cluster[i][1]-511),2)+ math.pow((dark_cluster[i][2]-511),2), .5)*c
        n_clusters_=2

        new_clusters = KMeans(n_clusters=n_clusters_, random_state=0).fit(dark_cluster_features)
        new_clusters_labels = new_clusters.labels_

        new_clusters_centers = new_clusters.cluster_centers_
        new_clusters_distance = [dist[2] for dist in new_clusters_centers]
        max_dist_cluster_label = np.argmax(new_clusters_distance)
        
        if max(new_clusters_distance) < min_dist:

            if os.path.exists(target1_image):
                seg_im = io.imread(target1_image)
                seg_imarray = numpy.array(seg_im)
            elif os.path.exists(target2_image):
                seg_im = io.imread(target2_image)
                seg_imarray = numpy.array(seg_im)[:,:,3]

            mean_pixel = np.mean(imarray)

            i = 0
            j = 0

            max_dist_cluster_total = 0

            segmented_image_total = 0
            segmented_noise_total = 0

            for x in range(imarray.shape[0]):
                for y in range(imarray.shape[1]):
                    cur_lab = labels[i]
                    if seg_imarray[x,y] > 127:
                        segmented_image_total +=1
                    if cur_lab == dark_cluster_label:
                        cur_label = new_clusters_labels[j]
                        if cur_label == max_dist_cluster_label:
                            max_dist_cluster_total += 1
                            if seg_imarray[x,y] > 127:
                                segmented_noise_total += 1
                            imarray[x,y]=replacement_color
                        new_image[x,y]=int((cur_label+1)*scale)
                        j+=1
                    else:
                        new_image[x,y]=int(30)
                    i+=1

            segmented_image_percentage = 1-float(segmented_noise_total/segmented_image_total)
            percentage_noise_removed = float(max_dist_cluster_total/(1024*1024))
            segmentation_total += segmented_image_total
            noise_removed += max_dist_cluster_total
            segmented_noise_overall += segmented_noise_total
            net_noise_removed += max_dist_cluster_total-segmented_noise_total
            io.imsave("vessels/val_transformed/m"+file, imarray)
            total_modified_images += 1

    return (net_noise_removed, noise_removed, segmented_noise_overall, total_modified_images)



def two_stage_kmeans(file_indices):
    path = os.path.join('vessels', 'train_inputs_transformed')
    files = glob.glob(path+"/*")

    for f in files:
        os.remove(f)


    noise_removed = 0
    segmented_noise_overall = 0
    net_noise_removed = 0
    image_map = dict()
    segmentation_total = 0
    total_modified_images = 0
    files_list = os.listdir(os.path.join('vessels', 'inputs'))

    min_dist_cand = [1024 * pow(2,.5)]
    min_dist_thresholds = []
    for file_index in file_indices:
        
        file = files_list[file_index]
        new_path = os.path.join('vessels', 'train_inputs_transformed', file)

        input_image = os.path.join('vessels', 'inputs', file)
        copyfile(input_image, new_path)
        
        target1_image = os.path.join('vessels', 'targets1', file)
        target2_image = os.path.join('vessels', 'targets2', file)
        
        file_loc = input_image
        max_dist = 0
        
        im = io.imread(file_loc)
        imarray = numpy.array(im)
        im_features = np.zeros((imarray.shape[0]*imarray.shape[1],1))
        i=0
        for x in range(imarray.shape[0]):
            for y in range(imarray.shape[1]):
                im_features[i,0]=imarray[x,y]
                i+=1
        print(im_features.shape)

        n_clusters_=2

        db = KMeans(n_clusters=n_clusters_, random_state=0).fit(im_features)
        labels = db.labels_
        centers = db.cluster_centers_

        dark_cluster_label = np.argmin(centers)
        replacement_color = np.max(centers)
        
        print(dark_cluster_label)
        print(n_clusters_)
        print(centers)

        new_image = np.zeros((imarray.shape[0], imarray.shape[1]), dtype=np.int8)

        scale = 255/3

        dark_cluster = []
        i = 0
        for x in range(imarray.shape[0]):
            for y in range(imarray.shape[1]):
                cur_lab = labels[i]
                if cur_lab == dark_cluster_label:
                    dark_cluster.append([i,x,y,imarray[x,y]])
                i+=1

        length_dark_cluster = len(dark_cluster)
        dark_cluster_features = np.zeros((length_dark_cluster,3))

        a = 1
        b = 1
        c = 2

        for i in range(length_dark_cluster):
            dark_cluster_features[i,0]= dark_cluster[i][1]*a
            dark_cluster_features[i,1]= dark_cluster[i][2]*b
            dark_cluster_features[i,2]= math.pow(math.pow((dark_cluster[i][1]-511),2)+ math.pow((dark_cluster[i][2]-511),2), .5)*c
        n_clusters_=2

        new_clusters = KMeans(n_clusters=n_clusters_, random_state=0).fit(dark_cluster_features)
        new_clusters_labels = new_clusters.labels_

        new_clusters_centers = new_clusters.cluster_centers_
        new_clusters_distance = [dist[2] for dist in new_clusters_centers]
        max_dist_cluster_label = np.argmax(new_clusters_distance)
        if os.path.exists(target1_image):
            seg_im = io.imread(target1_image)
            seg_imarray = numpy.array(seg_im)
        elif os.path.exists(target2_image):
            seg_im = io.imread(target2_image)
            seg_imarray = numpy.array(seg_im)[:,:,3]
        i = 0
        j = 0

        max_dist_cluster_total = 0

        segmented_image_total = 0
        segmented_noise_total = 0

        mean_pixel = np.mean(imarray)
        alt_image = np.copy(imarray)        

        for x in range(imarray.shape[0]):
            for y in range(imarray.shape[1]):
                cur_lab = labels[i]
                if seg_imarray[x,y] > 127:
                    segmented_image_total +=1
                if cur_lab == dark_cluster_label:
                    cur_label = new_clusters_labels[j]
                    if cur_label == max_dist_cluster_label:
                        max_dist_cluster_total += 1
                        if seg_imarray[x,y] > 127:
                            segmented_noise_total += 1
                        imarray[x,y]=replacement_color
                        alt_image[x,y] = 255
                    new_image[x,y]=int((cur_label+1)*scale)
                    j+=1
                else:
                    new_image[x,y]=int(30)
                i+=1
        segmented_image_percentage = 1-float(segmented_noise_total/segmented_image_total)
        percentage_noise_removed = float(max_dist_cluster_total/(1024*1024))
        segmentation_total += segmented_image_total
        if segmented_image_percentage > .98:
            noise_removed += max_dist_cluster_total
            segmented_noise_overall += segmented_noise_total
            net_noise_removed += max_dist_cluster_total-segmented_noise_total
            io.imsave("vessels/train_inputs_transformed/m"+file, imarray)
            io.imsave("vessels/train_inputs_transformed/a"+file, alt_image)
            total_modified_images += 1
            min_dist_cand.append(max(new_clusters_distance))
        else:
            min_dist_thresholds.append(max(new_clusters_distance))

        min_dist_cand_copy = copy.copy(min_dist_cand)
        for threshold in min_dist_thresholds:
            remove_list =[]
            for min_dist in min_dist_cand_copy:
                if min_dist < threshold:
                   min_dist_cand.remove(min_dist)
                   remove_list.append(min_dist)
            for remove_it in remove_list:
                min_dist_cand_copy.remove(remove_it)

        min_dist = min(min_dist_cand)
                                
    return (min_dist, net_noise_removed, noise_removed, segmented_noise_overall, total_modified_images, segmentation_total/len(file_indices))

test_clustering()
