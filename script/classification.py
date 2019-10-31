import numpy as np
import tensorflow as tf
import csv
import time
import os
import progressbar
import random
import matplotlib.pyplot as plt
from model import pointnet

RANDOM_SEED = 1234
flag = tf.app.flags

flag.DEFINE_integer('batch_size', 16, 'Batch size to use during training.')
flag.DEFINE_float('learning_rate', 1e-3, 'Learning rate.')
flag.DEFINE_integer('n_hidden', 64, 'Size of each model layer.')
flag.DEFINE_integer('num_row', 28, 'number of rows.')
flag.DEFINE_integer('num_col', 28, 'number of columns.')
flag.DEFINE_integer('num_pt', 28**2, 'max num of point.')
flag.DEFINE_integer('dim_input', 3, 'Size of input.')
flag.DEFINE_integer('num_class', 10, 'Number of classes.')
flag.DEFINE_string('data_dir', '/Work/data/mnist', 'Data directory')
flag.DEFINE_string('model_dir', '/Work/git/3D/subsampling', 'saved model directory.')
flag.DEFINE_integer('max_epoch', 300, 'max epochs.')
flag.DEFINE_boolean('save_model', False, 'save model.')
flag.DEFINE_boolean('load_model', False, 'load model.')
flag.DEFINE_boolean('is_training', False, 'training or not.')
flag.DEFINE_string('model_name', 'test', 'model name.')
flags = flag.FLAGS

def read_data(data_path):
    with open(data_path, 'rb') as csvfile:
        data = np.stack(list(csv.reader(csvfile, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)), axis=0)
    # convert to 2d point cloud
    x = data[:, 1:].astype(np.int32)
    y = data[:, 0].astype(np.int32)
    data = []
    for image_1d, category in zip(x, y):
        image_2d = np.reshape(image_1d, [flags.num_row, flags.num_col])
        # show_img(image_2d)
        rows, columns = np.where(image_2d >= 0)
        val = image_2d[rows, columns]/255.
        points_xyv = np.stack([rows, columns, val], axis=1).astype(np.float32)
        # show_points_as_img(points_xyv)
        data.append([points_xyv, category])
    return data

def show_points_as_img(points_xyv):
    image_2d = np.zeros([flags.num_row, flags.num_col], dtype=np.float32)
    image_2d[points_xyv[:, 0].astype(np.int32), points_xyv[:, 1].astype(np.int32)] = points_xyv[:, 2]
    plt.figure('point 2 image')
    plt.imshow(image_2d, cmap="gray")
    plt.pause(0.1)

def show_img(image_2d):
    plt.figure('raw image')
    plt.imshow(image_2d, cmap="gray")
    plt.pause(0.1)

def get_a_batch(data, start):
    batch_x = []
    batch_y = []
    for i in xrange(flags.batch_size):
        sample = data[min(start+i, len(data)-1)]
        points = sample[0]
        np.random.shuffle(points)
        # show_points_as_img(points)
        batch_x.append(points)
        batch_y.append(sample[1])
    return np.stack(batch_x), np.stack(batch_y)

def training(sess):
    train_data = read_data(os.path.join(flags.data_dir, 'mnist_train.csv'))
    valid_data = read_data(os.path.join(flags.data_dir, 'mnist_test.csv'))
    batch_size = flags.batch_size
    model = pointnet(sess, flags)
    init_temp = 1.
    final_temp = 0.01
    decay = (final_temp/init_temp)**(1./float(flags.max_epoch))

    trainable_var = tf.trainable_variables()
    part_var = []
    print('Trainable var list: ')
    for idx, v in enumerate(trainable_var):
        print( '  var {:3}: {:20}   {}'.format(idx, str(v.get_shape()), v.name))

    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init_op)

    model_dir = os.path.join(flags.model_dir, flags.model_name)
    saver = tf.train.Saver(max_to_keep=1, save_relative_paths=True)
    if flags.load_model:
        checkpoint = tf.train.get_checkpoint_state(model_dir) 
        if checkpoint and checkpoint.model_checkpoint_path:
            saver.restore(sess, checkpoint.model_checkpoint_path)
            print( 'model loaded: ', checkpoint.model_checkpoint_path )
        else:
            print( 'model not found')

    start_time = time.time()
    print( 'start training')
    temp = init_temp
    for epoch in range(flags.max_epoch):
        # training
        loss_list = []
        acc_list = []
        opt_time = []
        end_flag = False
        random.shuffle(train_data)
        bar = progressbar.ProgressBar(maxval=len(train_data)/batch_size+len(valid_data)/batch_size, \
                                      widgets=[progressbar.Bar('=', '[', ']'), ' ', 
                                               progressbar.Percentage()])
        all_t = 0
        for t in xrange(len(train_data)/batch_size):
            batch_data = get_a_batch(train_data, t*batch_size)
            acc, loss, _ = model.train(batch_data, temp)
            loss_list.append(loss)
            acc_list.append(acc)
            all_t += 1
            bar.update(all_t)
        loss_train = np.mean(loss_list)
        acc_train = np.mean(acc_list)

        # validating
        loss_list = []
        acc_list = []
        end_flag = False
        pos = 0
        for t in xrange(len(valid_data)/batch_size):
            batch_data = get_a_batch(valid_data, t*batch_size)
            acc, loss = model.validate(batch_data, temp)
            loss_list.append(loss)
            acc_list.append(acc)
            all_t += 1
            bar.update(all_t)
        bar.finish()
        loss_valid = np.mean(loss_list)
        acc_valid = np.mean(acc_list)

        info_train = '| Epoch:{:3d}'.format(epoch) + \
                     '| TrainLoss: {:2.5f}'.format(loss_train) + \
                     '| TestLoss: {:2.5f}'.format(loss_valid) + \
                     '| TrainAcc: {:2.5f}'.format(acc_train) + \
                     '| TestAcc: {:2.5f}'.format(acc_valid) + \
                     '| Time(min): {:2.1f}'.format((time.time() - start_time)/60.) + \
                     '| Temp: {:1.5f}'.format(temp)
        print(info_train)
        temp *= decay
        if flags.save_model and epoch == flags.max_epoch-1:
            saver.save(sess, os.path.join(model_dir, 'network') , global_step=epoch)

if __name__ == '__main__':
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        training(sess)