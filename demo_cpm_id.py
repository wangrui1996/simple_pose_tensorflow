# For single hand and no body part in the picture
# ======================================================

import tensorflow as tf
from models.nets import cpm_hand_slim, cpm_hand
import numpy as np
from utils import cpm_utils
import cv2
import time
import math
import sys

"""Parameters
"""
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('DEMO_TYPE',
                           default_value='test_imgs/people2.jpg',
                           # default_value='SINGLE',
                           docstring='MULTI: show multiple stage,'
                                     'SINGLE: only last stage,'
                                     'HM: show last stage heatmap,'
                                     'paths to .jpg or .png image')
tf.app.flags.DEFINE_string('model_path',
                           default_value='models/weights/cpm_hand.pkl',
                           docstring='Your model')
tf.app.flags.DEFINE_integer('cmap_radius',
                            default_value=3,
                            docstring='Center map gaussian variance')
tf.app.flags.DEFINE_integer('stages',
                            default_value=3,
                            docstring='How many CPM stages')
tf.app.flags.DEFINE_integer('cam_num',
                            default_value=0,
                            docstring='Webcam device number')

# Set color for each finger
joint_color_code = [[139, 53, 255],
                    [0, 56, 255],
                    [43, 140, 237],
                    [37, 168, 36],
                    [147, 147, 0],
                    [70, 17, 145]]

if sys.version_info.major == 3:
    PYTHON_VERSION = 3
else:
    PYTHON_VERSION = 2


def main(argv):
    flags_joints = 6
    flags_stages = 3
    flags_input_size = 32*8
    flags_hmap_size = 32
    flags_color_channel = 'RGB'
    tf_device = '/cpu:0'
    with tf.device(tf_device):
        """Build graph
        """
        if flags_color_channel == 'RGB':
            input_data = tf.placeholder(dtype=tf.float32, shape=[None, flags_input_size, flags_input_size, 3],
                                        name='input_image')
        else:
            input_data = tf.placeholder(dtype=tf.float32, shape=[None, flags_input_size, flags_input_size, 1],
                                        name='input_image')

        center_map = tf.placeholder(dtype=tf.float32, shape=[None, flags_input_size, flags_input_size, 1],
                                    name='center_map')

        model = cpm_hand.CPM_Model(flags_input_size, flags_hmap_size,flags_stages,flags_joints ,input_image= input_data)
        #cpm_hand_slim.CPM_Model
        #model.build_model(input_data, center_map, 1)

    saver = tf.train.Saver()

    """Create session and restore weights
    """
    sess = tf.Session()
    FLAGS.model_path = "/Users/wangrui/work/convolutional-pose-machines-tensorflow/utils/init_0.001_rate_0.5_step_10000/cpm_hand-1000"
    sess.run(tf.global_variables_initializer())
    if FLAGS.model_path.endswith('pkl'):
        model.load_weights_from_file(FLAGS.model_path, sess, False)
    else:
        saver.restore(sess, FLAGS.model_path)


    test_center_map = cpm_utils.gaussian_img(flags_input_size, flags_input_size, flags_input_size / 2,
                                             flags_input_size / 2,
                                             FLAGS.cmap_radius)
    test_center_map = np.reshape(test_center_map, [1, flags_input_size, flags_input_size, 1])

    # Check weights
    for variable in tf.trainable_variables():
        with tf.variable_scope('', reuse=True):
            var = tf.get_variable(variable.name.split(':0')[0])
            print(variable.name, np.mean(sess.run(var)))

    if not FLAGS.DEMO_TYPE.endswith(('png', 'jpg')):
        cam = cv2.VideoCapture(FLAGS.cam_num)

    with tf.device(tf_device):

        while True:
            t1 = time.time()
            if FLAGS.DEMO_TYPE.endswith(('png', 'jpg')):
                test_img = cpm_utils.read_image(FLAGS.DEMO_TYPE, [], flags_input_size, 'IMAGE')
            else:
                test_img = cpm_utils.read_image([], cam, flags_input_size, 'WEBCAM')

            test_img_resize = cv2.resize(test_img, (flags_input_size, flags_input_size))
            print('img read time %f' % (time.time() - t1))


            test_img_input = test_img_resize / 255.0 - 0.5
            test_img_input = np.expand_dims(test_img_input, axis=0)


            if FLAGS.DEMO_TYPE.endswith(('png', 'jpg')):
                # Inference
                t1 = time.time()
                predict_heatmap, stage_heatmap_np = sess.run([model.current_heatmap,
                                                              model.stage_heatmap,
                                                              ],
                                                             feed_dict={'input_image:0': test_img_input,
                                                                        'center_map:0': test_center_map})

                # Show visualized image
                demo_img = visualize_result(test_img, stage_heatmap_np, flags_hmap_size, flags_joints)
                cv2.imshow('demo_img', demo_img.astype(np.uint8))
                if cv2.waitKey(0) == ord('q'): break
                print('fps: %.2f' % (1 / (time.time() - t1)))

def visualize_result(test_img, stage_heatmap_np, hmap_size, num_joints):
    t1 = time.time()
    demo_stage_heatmaps = []
    last_heatmap = stage_heatmap_np[len(stage_heatmap_np) - 1][0, :, :, 0:num_joints].reshape(
    (hmap_size, hmap_size, num_joints))
    last_heatmap = cv2.resize(last_heatmap, (test_img.shape[1], test_img.shape[0]))
    print('hm resize time %f' % (time.time() - t1))

    t1 = time.time()
    joint_coord_set = np.zeros((num_joints, 2))

    for joint_num in range(num_joints):
        joint_coord = np.unravel_index(np.argmax(last_heatmap[:, :, joint_num]),
                                           (test_img.shape[0], test_img.shape[1]))
        joint_coord_set[joint_num, :] = [joint_coord[0], joint_coord[1]]

        color_code_num = (joint_num // 4)
        if joint_num in [0, 4, 8, 12, 16]:
            if PYTHON_VERSION == 3:
                joint_color = list(map(lambda x: x + 35 * (joint_num % 4), joint_color_code[color_code_num]))
            else:
                joint_color = map(lambda x: x + 35 * (joint_num % 4), joint_color_code[color_code_num])

            cv2.circle(test_img, center=(joint_coord[1], joint_coord[0]), radius=3, color=joint_color, thickness=-1)
        else:
            if PYTHON_VERSION == 3:
                joint_color = list(map(lambda x: x + 35 * (joint_num % 4), joint_color_code[color_code_num]))
            else:
                joint_color = map(lambda x: x + 35 * (joint_num % 4), joint_color_code[color_code_num])

            cv2.circle(test_img, center=(joint_coord[1], joint_coord[0]), radius=3, color=joint_color, thickness=-1)
    print('plot joint time %f' % (time.time() - t1))

    t1 = time.time()
    # Plot limb colors
    return test_img


if __name__ == '__main__':
    tf.app.run()
