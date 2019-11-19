import numpy as np
import tensorflow as tf
import time
import os
import progressbar
import random
from model import network
from utils import *
from tensorboard.plugins.mesh import summary as mesh_summary

RANDOM_SEED = 1234
flag = tf.app.flags

flag.DEFINE_integer('batch_size', 16, 'Batch size to use during training.')
flag.DEFINE_float('learning_rate', 1e-3, 'Learning rate.')
flag.DEFINE_integer('n_hidden', 64, 'Size of each model layer.')
flag.DEFINE_integer('num_row', 28, 'number of rows.')
flag.DEFINE_integer('num_col', 28, 'number of columns.')
flag.DEFINE_integer('num_pt', 1024, 'max num of point.') # 28**2
flag.DEFINE_integer('dim_input', 3, 'Size of input.')
flag.DEFINE_integer('num_class', 40, 'Number of classes.')
flag.DEFINE_string('data_dir', '/Work/data/mnist', 'Data directory')
flag.DEFINE_string('model_dir', '/Work/git/3D/subsampling/saved_network', 'saved model directory.')
flag.DEFINE_integer('max_epoch', 100, 'max epochs.')
flag.DEFINE_boolean('save_model', True, 'save model.')
flag.DEFINE_boolean('load_model', False, 'load model.')
flag.DEFINE_boolean('is_training', True, 'training or not.')
flag.DEFINE_string('model_name', 'test', 'model name.')
flag.DEFINE_string('sample_mode', 'normal', '[uniform, normal, determine, concrete]')
flag.DEFINE_string('task', 'classification', '[reconstruction, classification]')
flags = flag.FLAGS

def training(sess):
    print('load data from: ', flags.data_dir)
    if 'mnist' in flags.data_dir:
        train_data = read_mnist_data(os.path.join(flags.data_dir, 'mnist_train.csv'))
        valid_data = read_mnist_data(os.path.join(flags.data_dir, 'mnist_test.csv'))
    elif 'modelnet40' in flags.data_dir:
        train_data, valid_data, _ = read_modelnet40_data(flags.data_dir, flags.num_pt)
    else:
        print('data path unrecognised!')
        return
    batch_size = flags.batch_size

    model = network(sess, flags)
    init_temp = 1.
    final_temp = 0.01
    decay = (final_temp/init_temp)**(1./float(flags.max_epoch))

    trainable_var = tf.trainable_variables()
    part_var = []
    print('Trainable var list: ')
    for idx, v in enumerate(trainable_var):
        print('  var {:3}: {:20}   {}'.format(idx, str(v.get_shape()), v.name))

    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init_op)

    model_dir = os.path.join(flags.model_dir, flags.model_name)
    if not os.path.exists(model_dir): 
        os.makedirs(model_dir)
    saver = tf.train.Saver(max_to_keep=1, save_relative_paths=True)
    if flags.load_model:
        checkpoint = tf.train.get_checkpoint_state(model_dir) 
        if checkpoint and checkpoint.model_checkpoint_path:
            saver.restore(sess, checkpoint.model_checkpoint_path)
            print('model loaded: ', checkpoint.model_checkpoint_path )
        else:
            print('model not found')
    summary_writer = tf.summary.FileWriter(model_dir, sess.graph)
    xyz_ph = tf.placeholder(tf.float32, shape=[flags.batch_size, flags.num_pt, 3], name='xyz_ph')
    rgb_ph = tf.placeholder(tf.float32, shape=[flags.batch_size, flags.num_pt, 3], name='rgb_ph')
    point_summary = mesh_summary.op('point_cloud', vertices=xyz_ph, colors=rgb_ph)
    train_acc_ph = tf.placeholder(tf.float32, shape=[], name='train_acc_ph')
    train_acc_summary = tf.summary.scalar('train_acc', train_acc_ph)
    test_acc_ph = tf.placeholder(tf.float32, shape=[], name='test_acc_ph')
    test_acc_summary = tf.summary.scalar('test_acc', test_acc_ph)
    temp_summary = tf.summary.scalar('temperature', model.t)
    merged = tf.summary.merge_all()

    start_time = time.time()
    print('start training')
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
        for t in range(int(len(train_data)/batch_size)):
            batch_data = get_a_batch(train_data, t*batch_size, batch_size)
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
        for t in range(int(len(valid_data)/batch_size)):
            batch_data = get_a_batch(valid_data, t*batch_size, batch_size)
            acc, loss, sampled_points, score = model.validate(batch_data, temp)
            loss_list.append(loss)
            acc_list.append(acc)
            all_t += 1
            bar.update(all_t)
        bar.finish()
        loss_valid = np.mean(loss_list)
        acc_valid = np.mean(acc_list)
        points_rgb = (np.stack([score, np.zeros_like(score), np.zeros_like(score)], axis=2)*255).astype(int)
        points_xyz = batch_data[0]

        info_train = '| Epoch:{:3d}'.format(epoch) + \
                     '| TrainLoss: {:2.5f}'.format(loss_train) + \
                     '| TestLoss: {:2.5f}'.format(loss_valid) + \
                     '| TrainAcc: {:2.5f}'.format(acc_train) + \
                     '| TestAcc: {:2.5f}'.format(acc_valid) + \
                     '| Time(min): {:2.1f}'.format((time.time() - start_time)/60.) + \
                     '| Temp: {:1.5f}'.format(temp)
        print(info_train)
        summary = sess.run(merged, feed_dict={xyz_ph: points_xyz,
                                              rgb_ph: points_rgb,
                                              train_acc_ph: acc_train,
                                              test_acc_ph: acc_valid,
                                              model.t: temp})
        summary_writer.add_summary(summary, epoch)
        if flags.save_model and epoch == flags.max_epoch-1:
            saver.save(sess, os.path.join(model_dir, 'network') , global_step=epoch)
        if flags.sample_mode in ['normal', 'determine', 'concrete']:
            temp *= decay

if __name__ == '__main__':
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        training(sess)