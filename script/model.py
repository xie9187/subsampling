import tf_util
import tensorflow as tf
import numpy as np

class pointnet(object):
    def __init__(self, sess, flags):
        self.sess = sess
        self.num_pt = flags.num_pt
        self.dim_input = flags.dim_input
        self.num_class = flags.num_class
        self.learning_rate = flags.learning_rate
        self.batch_size = flags.batch_size

        self.input = tf.placeholder(tf.float32, shape=[self.batch_size, self.num_pt, self.dim_input], name='input') # b, n, input
        self.label = tf.placeholder(tf.int32, shape=[self.batch_size], name='label') # b, label
        self.is_training = tf.placeholder(tf.bool, shape=[], name='is_training')

        # sampling
        sampled_input = uniform_sampling(self.input)

        logits = self.get_model(sampled_input, self.is_training)
        y = tf.argmax(logits, axis=1, output_type=tf.int32)
        self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.label, logits=logits)
        self.opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
        correct_pred = tf.equal(y, self.label)
        self.acc = tf.reduce_mean(tf.cast(correct_pred, tf.int32))

    def get_model(self, inputs, is_training, bn_decay=None):
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

    def train(self, batch_data):
        return self.sess.run([self.acc, self.loss, self.opt], feed_dict={
            self.input: batch_data[0],
            self.label: batch_data[1],
            self.is_training: True
            })

    def validate(self, batch_data):
        return self.sess.run([self.acc, self.loss], feed_dict={
            self.input: batch_data[0],
            self.label: batch_data[1],
            self.is_training: True
            })

def uniform_sampling(features, subset_size=100):
    shape = features.get_shape().as_list()
    coord1 = tf.reshape(tf.tile(tf.expand_dims(tf.range(shape[0]), axis=-1), [1, subset_size]), [-1])
    coord2 = tf.tile(tf.range(subset_size), [shape[0]])
    coords = tf.stack([coord1, coord2], axis=1) # b*sub_n, 2
    coords = tf.reshape(coords, [shape[0], subset_size, 2])
    sub_features = tf.gather_nd(features, coords) # b*sub_n, 2
    return sub_features

if __name__ == '__main__':
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        a = tf.placeholder(tf.float32, shape=[2, 10, 5], name='a')
        b = uniform_sampling(a, 3)
        data = np.reshape(np.arange(100, dtype=np.float32), [2, 10, 5])
        print(sess.run(b, feed_dict={a: data}))