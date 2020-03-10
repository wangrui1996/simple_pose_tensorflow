import cv2
import cpm_utils
import numpy as np
import math
import tensorflow as tf
import time
import json
import random
import os


tfr_file = 'cpm_sample_dataset.tfrecords'
dataset_dir = '/Users/wangrui/Downloads/id_dataset/data'

SHOW_INFO = False
box_size = 32
input_size = 256
num_of_joints = 6
gaussian_radius = 2


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _float64_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


# Create writer
tfr_writer = tf.python_io.TFRecordWriter(tfr_file)

img_count = 0
t1 = time.time()
images_dir = os.path.join(dataset_dir, "images")
annotations_dir = os.path.join(dataset_dir, "annotations")
# Loop each dir
for file_name in os.listdir(images_dir):

    image_path = os.path.join(images_dir, file_name)
    annotation_path = os.path.join(annotations_dir, "{}.json".format(file_name.split(".")[0]))

    #cur_img_path = dataset_dir + person_dir + '/imgs/' + line[0]
    cur_img = cv2.imread(image_path)
    print(image_path)
    inp_f = open(annotation_path, 'r')
    json_data = json.load(inp_f)
    #json_data["shapes"] = ""

    def get_bbox_and_joints_from_json(shapes):
        assert len(shapes) == 2  # must be len is 2, one is bbox and annother is text
        assert shapes[0]["label"] in ["zhen","fan","zheng","text"]
        assert shapes[1]["label"] in ["zhen","fan","zheng","text"]
        bbox_idx = 0
        if shapes[bbox_idx]["label"]=="text":
            bbox_idx = 1

        bbox_point = shapes[bbox_idx]["points"]
        bx_x1, bx_y1 = bbox_point[0]
        bx_x2, bx_y2 = bbox_point[2]
        cur_id_bbox = [min([bx_x1, bx_x2]),
                         min([bx_y1, bx_y2]),
                         max([bx_x1, bx_x2]),
                         max([bx_y1, bx_y2])]
        #if cur_hand_bbox[0] < 0: cur_hand_bbox[0] = 0
        #if cur_hand_bbox[1] < 0: cur_hand_bbox[1] = 0
        #if cur_hand_bbox[2] > cur_img.shape[1]: cur_hand_bbox[2] = cur_img.shape[1]
        #if cur_hand_bbox[3] > cur_img.shape[0]: cur_hand_bbox[3] = cur_img.shape[0]
        text_bx = shapes[1-bbox_idx]["points"]

        tmpx1,tmpy1 = text_bx[0]
        tmpx2,tmpy2 = text_bx[1]
        tmpx3,tmpy3 = text_bx[2]
        text_arr = np.array(text_bx).transpose()
        x_list = text_arr[0]
        y_list = text_arr[1]
        axis_1 = np.where(y_list==y_list.min())[0]
        axis_3 = np.where(x_list==x_list.max())[0]
        axis_2 = 3 - axis_1 - axis_3
        cur_id_joints_x = [-1 for _ in range(6)]
        cur_id_joints_y = [-1 for _ in range(6)]
        sub_add = 0
        is_zhen = True
        if shapes[bbox_idx]["label"] == "fan":
            is_zhen = False
            sub_add = 3

        cur_id_joints_x[sub_add] = x_list[axis_1][0]
        cur_id_joints_y[sub_add] = y_list[axis_1][0]
        cur_id_joints_x[sub_add+1] = x_list[axis_2][0]
        cur_id_joints_y[sub_add+1] = y_list[axis_2][0]
        cur_id_joints_x[sub_add+2] = x_list[axis_3][0]
        cur_id_joints_y[sub_add+2] = y_list[axis_3][0]
        return is_zhen, cur_id_bbox, cur_id_joints_x, cur_id_joints_y
    # Read in bbox and joints coords
    is_zhen, cur_id_bbox, cur_id_joints_x, cur_id_joints_y = get_bbox_and_joints_from_json(json_data["shapes"])
    print(cur_id_bbox)
    if is_zhen:
        gauss_range_list = [0, 1, 2]
    else:
        gauss_range_list = [3, 4, 5]
    #exit(0)

    #cur_hand_joints_x = [float(i) for i in line[9:49:2]]
    #cur_hand_joints_x.append(float(line[7]))
    #cur_hand_joints_y = [float(i) for i in line[10:49:2]]
    #cur_hand_joints_y.append(float(line[8]))

    # Crop image and adjust joint coords
    cur_img = cur_img[int(float(cur_id_bbox[1])):int(float(cur_id_bbox[3])),
                  int(float(cur_id_bbox[0])):int(float(cur_id_bbox[2])),
                  :]

    #cv2.imshow("demo", cur_img)
    cv2.imwrite("demo.jpg", cur_img)
    #cv2.waitKey(0)
    #exit(0)
    cur_id_joints_x = [x - cur_id_bbox[0] for x in cur_id_joints_x]
    cur_id_joints_y = [x - cur_id_bbox[1] for x in cur_id_joints_y]

    # # Display joints
    # for i in range(len(cur_hand_joints_x)):
    #     cv2.circle(cur_img, center=(int(cur_hand_joints_x[i]), int(cur_hand_joints_y[i])),radius=3, color=(255,0,0), thickness=-1)
    #     cv2.imshow('', cur_img)
    #     cv2.waitKey(500)
    # cv2.imshow('', cur_img)
    # cv2.waitKey(1)

    output_image = np.ones(shape=(input_size, input_size, 3)) * 128
    output_heatmaps = np.zeros((box_size, box_size, num_of_joints))

    # Resize and pad image to fit output image size
    if cur_img.shape[0] > cur_img.shape[1]:
        scale = input_size / (cur_img.shape[0] * 1.0)

        # Relocalize points
        cur_id_joints_x = map(lambda x: x * scale, cur_id_joints_x)
        cur_id_joints_y = map(lambda x: x * scale, cur_id_joints_y)

        # Resize image
        image = cv2.resize(cur_img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LANCZOS4)
        offset = image.shape[1] % 2

        output_image[:, int(input_size / 2 - math.floor(image.shape[1] / 2)): int(
                input_size / 2 + math.floor(image.shape[1] / 2) + offset), :] = image
        cur_id_joints_x = map(lambda x: x + (input_size / 2 - math.floor(image.shape[1] / 2)),
                                    cur_id_joints_x)
        scale = box_size / (cur_img.shape[0] * 1.0)
        # Relocalize points
        cur_id_joints_x = map(lambda x: x * scale, cur_id_joints_x)
        cur_id_joints_y = map(lambda x: x * scale, cur_id_joints_y)
        cur_id_joints_x = np.asarray(list(cur_id_joints_x))
        cur_id_joints_y = np.asarray(list(cur_id_joints_y))

        if SHOW_INFO:
            hmap = np.zeros((box_size, box_size))
                # Plot joints
            for i in range(num_of_joints):
                cv2.circle(output_image, (int(cur_id_joints_x[i]), int(cur_id_joints_y[i])), 3, (0, 255, 0), 2)

                # Generate joint gaussian map

                part_heatmap= cpm_utils.gaussian_img(box_size,box_size,cur_id_joints_x[i],cur_id_joints_y[i],1)
                #part_heatmap = utils.make_gaussian(output_image.shape[0], gaussian_radius,
                     #                                  [cur_hand_joints_x[i], cur_hand_joints_y[i]])
                hmap += part_heatmap * 50
        else:
            for i in range(num_of_joints):
                    #output_heatmaps[:, :, i] = utils.make_gaussian(box_size, gaussian_radius,
                    #                                               [cur_hand_joints_x[i], cur_hand_joints_y[i]])
                if i in gauss_range_list:
                        output_heatmaps[:, :, i]= cpm_utils.gaussian_img(box_size,box_size,cur_id_joints_x[i],cur_id_joints_y[i],1)

    else:
        scale = input_size / (cur_img.shape[1] * 1.0)

        # Relocalize points
        cur_id_joints_x = map(lambda x: x * scale, cur_id_joints_x)
        cur_id_joints_y = map(lambda x: x * scale, cur_id_joints_y)

        # Resize image
        image = cv2.resize(cur_img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LANCZOS4)
        offset = image.shape[0] % 2
        output_image[int(input_size / 2 - math.floor(image.shape[0] / 2)): int(
                input_size / 2 + math.floor(image.shape[0] / 2) + offset), :, :] = image
        cur_id_joints_y = map(lambda x: x + (input_size / 2 - math.floor(image.shape[0] / 2)),
                                    cur_id_joints_y)
        scale = box_size / (input_size * 1.0)
        # Relocalize points
        cur_id_joints_x = map(lambda x: x * scale, cur_id_joints_x)
        cur_id_joints_y = map(lambda x: x * scale, cur_id_joints_y)
        cur_id_joints_x = np.asarray(list(cur_id_joints_x))
        cur_id_joints_y = np.asarray(list(cur_id_joints_y))

        if SHOW_INFO:
            hmap = np.zeros((box_size, box_size))
            # Plot joints
            for i in range(num_of_joints):
                cv2.circle(output_image, (int(cur_id_joints_x[i]), int(cur_id_joints_y[i])), 3, (0, 255, 0), 2)

                # Generate joint gaussian map
                #part_heatmap = cpm_utils.make_gaussian(output_image.shape[0], gaussian_radius,
                 #                                      [cur_id_joints_x[i], cur_id_joints_y[i]])
                #hmap += part_heatmap * 50
            cv2.imshow("demo", output_image)
            cv2.waitKey(0)
        else:
            for i in range(num_of_joints):
                if i in gauss_range_list:
                    output_heatmaps[:, :, i] = cpm_utils.make_gaussian(box_size, gaussian_radius,
                                                                   [cur_id_joints_x[i], cur_id_joints_y[i]])
    if SHOW_INFO:
        cv2.imshow('', hmap.astype(np.uint8))
        cv2.imshow('i', output_image.astype(np.uint8))
        cv2.waitKey(0)

    # Create background map
    output_background_map = np.ones((box_size, box_size)) - np.amax(output_heatmaps, axis=2)
    output_heatmaps = np.concatenate((output_heatmaps, output_background_map.reshape((box_size, box_size, 1))),
                                         axis=2)
    # cv2.imshow('', (output_background_map*255).astype(np.uint8))
    # cv2.imshow('h', (np.amax(output_heatmaps[:, :, 0:21], axis=2)*255).astype(np.uint8))
    # cv2.waitKey(1000)


    coords_set = np.concatenate((np.reshape(cur_id_joints_x, (num_of_joints, 1)),
                                     np.reshape(cur_id_joints_y, (num_of_joints, 1))),
                                    axis=1)
    output_image_raw = output_image.astype(np.uint8).tostring()
    output_heatmaps_raw = output_heatmaps.flatten().tolist()
    output_coords_raw = coords_set.flatten().tolist()

    raw_sample = tf.train.Example(features=tf.train.Features(feature={
            'image': _bytes_feature(output_image_raw),
            'heatmaps': _float64_feature(output_heatmaps_raw)
        }))

    tfr_writer.write(raw_sample.SerializeToString())

    img_count += 1
    if img_count % 50 == 0:
            print('Processed %d images, took %f seconds' % (img_count, time.time() - t1))
            t1 = time.time()

tfr_writer.close()
