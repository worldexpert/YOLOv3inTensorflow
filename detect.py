# reference - https://github.com/mystic123/tensorflow-yolo-v3.git
from utils import *
import time

import yolo_v3
import yolo_v3_tiny

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string(
    'input_img', '/home/rei/workspace/worldexpert/YOLOv3inTensorflow/4145_23325_2526.jpg', 'Input image')
tf.app.flags.DEFINE_string(
    'output_img', '/home/rei/workspace/worldexpert/YOLOv3inTensorflow/result.jpg', 'Output image')
tf.app.flags.DEFINE_integer(
    'size', 416, 'Image size')

tf.app.flags.DEFINE_string(
    'class_names', 'coco.names', 'File with class names')
tf.app.flags.DEFINE_string(
    #'frozen_model', '', 'Frozen tensorflow protobuf model')
    'frozen_model', '/home/rei/workspace/worldexpert/YOLOv3inTensorflow/frozen_darknet_yolov3_model.pb', 'Frozen tensorflow protobuf model')

def main(argv=None):

    #when sharing GPU, you can allocate GPU memory with the fraction
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction= 1.0)

    # To find out which devices your operations and tensors are assigned to,
    # create the session with log_device_placement configuration option set to True.
    config = tf.ConfigProto(gpu_options= gpu_options, log_device_placement=False)

    img = Image.open(FLAGS.input_img)
    img_resized = letter_box_image(img, FLAGS.size, FLAGS.size, 128)
    img_resized = img_resized.astype(np.float32)
    classes = load_coco_names(FLAGS.class_names)

    if FLAGS.frozen_model:

        t0 = time.time()
        frozenGraph = load_graph(FLAGS.frozen_model)
        print("Loaded graph in {:.2f}s".format(time.time()-t0))

        boxes, inputs = get_boxes_and_inputs_pb(frozenGraph)

        with tf.Session(graph=frozenGraph, config=config) as sess:
            t0 = time.time()
            #shape : batch_size x 10647 x (num_classes + 5 bounding box attrs)
            #The number 10647 is equal to the sum 507 +2028 + 8112, which are the numbers of possible objects detected on each scale.
            #The 5 values describing bounding box attributes stand for center_x, center_y, width, height
            detected_boxes = sess.run(
                boxes, feed_dict={inputs: [img_resized]})
    '''
    else:
        if FLAGS.tiny:
            model = yolo_v3_tiny.yolo_v3_tiny
        else:
            model = yolo_v3.yolo_v3

        boxes, inputs = get_boxes_and_inputs(model, len(classes), FLAGS.size, FLAGS.data_format)

        saver = tf.train.Saver(var_list=tf.global_variables(scope='detector'))

        with tf.Session(config=config) as sess:
            t0 = time.time()
            saver.restore(sess, FLAGS.ckpt_file)
            print('Model restored in {:.2f}s'.format(time.time()-t0))

            t0 = time.time()
            detected_boxes = sess.run(
                boxes, feed_dict={inputs: [img_resized]})
    '''
    filtered_boxes = non_max_suppression(detected_boxes)
    print("detected_boxes ", detected_boxes)
    print("filtered_boxes ", filtered_boxes)
    print("Predictions found in {:.2f}s".format(time.time() - t0))

    draw_boxes(filtered_boxes, img, classes, (FLAGS.size, FLAGS.size), True)

    img.save(FLAGS.output_img)

if __name__ == '__main__':
    tf.app.run()