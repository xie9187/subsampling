import tf_util
import model_utils
import tensorflow as tf
import numpy as np

class network(object):
    def __init__(self, sess, flags):
        self.sess = sess
        self.num_pt = flags.num_pt
        self.dim_input = flags.dim_input
        self.dim_fpfh = flags.fpfh
        self.num_class = flags.num_class
        self.learning_rate = flags.learning_rate
        self.batch_size = flags.batch_size
        self.task = flags.task

        self.input = tf.placeholder(tf.float32, shape=[self.batch_size, self.num_pt, self.dim_input], name='input') # b, n, input
        self.fpfh = tf.placeholder(tf.float32, shape=[self.batch_size, self.num_pt, self.dim_fpfh], name='fpfh') # b, n, input
        self.label = tf.placeholder(tf.int32, shape=[self.batch_size], name='label') # b, label
        self.is_training = tf.placeholder(tf.bool, shape=[], name='is_training')
        self.t = tf.placeholder(tf.float32, shape=[], name='temperature')


        # sampling
        if 'uniform' in flags.sample_mode:
            self.sampled_input = uniform_sampling(self.input, int(self.num_pt/4))
            self.score = tf.zeros([self.batch_size, self.num_pt])
        elif 'normal' in flags.sample_mode:
            self.sampled_input, self.score = normal_sampling(self.input, self.t, int(self.num_pt/4))
        elif 'concrete' in flags.sample_mode:
            self.sampled_input, self.score = concrete_sampling(self.input, self.t, int(self.num_pt/4))
        else:
            self.sampled_input = self.input
            self.score = tf.zeros([self.batch_size, self.num_pt])
        if flags.task == 'classification':
            logits = self.pointnet(self.sampled_input, self.is_training)
            y = tf.argmax(logits, axis=1, output_type=tf.int32)
            self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.label, logits=logits)
            correct_pred = tf.equal(y, self.label)
            self.acc = tf.reduce_mean(tf.cast(correct_pred, tf.int32))
        elif flags.task == 'reconstruction':
            pass

        self.opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)


    def pointnet(self, inputs, is_training, bn_decay=None):
        """ Classification PointNet, input is BxNx3, output Bx10 """
        shape = inputs.get_shape().as_list()
        batch_size = shape[0]
        num_pt = shape[1]
        input_image = tf.expand_dims(inputs, -1)
         
        # Point functions (MLP implemented as conv2d)
        net = tf_util.conv2d(input_image, 64, [1,3],
                             padding='VALID', stride=[1,1],
                             bn=True, is_training=is_training,
                             scope='conv1', bn_decay=bn_decay)
        net = tf_util.conv2d(net, 64, [1,1],
                             padding='VALID', stride=[1,1],
                             bn=True, is_training=is_training,
                             scope='conv2', bn_decay=bn_decay)
        net = tf_util.conv2d(net, 64, [1,1],
                             padding='VALID', stride=[1,1],
                             bn=True, is_training=is_training,
                             scope='conv3', bn_decay=bn_decay)
        net = tf_util.conv2d(net, 128, [1,1],
                             padding='VALID', stride=[1,1],
                             bn=True, is_training=is_training,
                             scope='conv4', bn_decay=bn_decay)
        net = tf_util.conv2d(net, 1024, [1,1],
                             padding='VALID', stride=[1,1],
                             bn=True, is_training=is_training,
                             scope='conv5', bn_decay=bn_decay)

        # Symmetric function: max pooling
        net = tf_util.max_pool2d(net, [num_pt,1],
                             padding='VALID', scope='maxpool')
        # MLP on global point cloud vector
        net = tf.reshape(net, [batch_size, -1])
        net = tf_util.fully_connected(net, 512, bn=True, is_training=is_training,
                                      scope='fc1', bn_decay=bn_decay)
        net = tf_util.fully_connected(net, 256, bn=True, is_training=is_training,
                                      scope='fc2', bn_decay=bn_decay)
        net = tf_util.dropout(net, keep_prob=0.7, is_training=is_training,
                              scope='dp1')
        net = tf_util.fully_connected(net, self.num_class, activation_fn=None, scope='fc3')

        return net

    def train(self, batch_data, t):
        return self.sess.run([self.acc, self.loss, self.opt], feed_dict={
            self.input: batch_data[0],
            self.label: batch_data[1],
            self.is_training: True,
            self.t: t
            })

    def validate(self, batch_data, t):
        return self.sess.run([self.acc, self.loss, self.sampled_input, self.score], feed_dict={
            self.input: batch_data[0],
            self.label: batch_data[1],
            self.is_training: True,
            self.t: t
            })

def uniform_sampling(features, k=100):
    b, n, d = features.get_shape().as_list()
    coord1 = tf.reshape(tf.tile(tf.expand_dims(tf.range(b), axis=-1), [1, k]), [-1])
    coord2 = tf.tile(tf.range(k), [b])
    coords = tf.reshape(tf.stack([coord1, coord2], axis=1), [b, k, 2]) # b, k, 2
    sub_features = tf.gather_nd(features, coords) # b, k, d
    return sub_features

def concrete_sampling(features, t, k=100):
    b, n, d = features.get_shape().as_list() # b, n, d
    alpha_h = model_utils.dense_layer(tf.reshape(features, [-1, d]), 256, 'alpha_h') # b*n, 256
    alpha = model_utils.dense_layer(alpha_h, 1, 'alpha', activation=None) # b*n, 1
    alpha_n = tf.tile(tf.reshape(alpha, [b, 1, n]), [1, k, 1]) # b, k, n
    uniform_noise = tf.random_uniform([b, k, n]) # b, k, n
    gumble_noise = -tf.log(-tf.log(uniform_noise)) # b, k, n
    noisy_alpha = (alpha_n + gumble_noise)/(t*10.)
    samples = tf.nn.softmax(noisy_alpha, axis=-1) # b, k, n
    sub_features = tf.matmul(samples, features) # b, k, d
    return sub_features, tf.reshape(alpha, [b, n])

def normal_sampling(features, t, k=100):
    b, n, d = features.get_shape().as_list() # b, n, d
    # score for each point
    score_h1 = model_utils.dense_layer(tf.reshape(features, [-1, d]), 256, 'score_h1') # b*n, 256
    origin_score = model_utils.dense_layer(score_h1, 1, 'score', activation=tf.nn.sigmoid) # b*n, 1
    score = tf.reshape(origin_score, [b, n]) # b, n
    noise = tf.nn.relu(tf.random.truncated_normal([b, n], stddev=t**2)) # b, n
    score += noise # b, n
    # sort with top_k
    sorted_score, sorted_indicies = tf.nn.top_k(score, n) # b, n
    coord1 = tf.reshape(tf.tile(tf.expand_dims(tf.range(b), axis=-1), [1, n]), [-1]) # b*k
    coord2 = tf.reshape(sorted_indicies, [-1]) # b*n
    coords = tf.reshape(tf.stack([coord1, coord2], axis=1), [b, n, 2]) # b, n, 2 
    sorted_features = tf.gather_nd(features, coords) # b, n, d
        
    top_features, bot_features = tf.split(sorted_features, [k, n-k], axis=1)
    top_scores, bot_scores = tf.split(sorted_score, [k, n-k], axis=1)

    # sampled features
    top_scores = tf.tile(tf.expand_dims(top_scores, axis=2), [1, 1, d]) # b, k, d
    # sub_features = tf.pow(top_scores, t) * top_features
    sub_features = top_scores * top_features
    return sub_features, tf.reshape(score, [b, n])

def reinforce_sampling(features):
    pass



if __name__ == '__main__':
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        a = tf.placeholder(tf.float32, shape=[2, 10, 5], name='a')
        b = uniform_sampling(a, 3)
        c, t = my_sampling(a, 3)
        data = np.reshape(np.arange(100, dtype=np.float32), [2, 10, 5])
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init_op)  
        b_val, c_val = sess.run([b, c], feed_dict={a: data, t: 1.})
        print(c_val)