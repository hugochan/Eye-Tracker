import os
import argparse
import timeit
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


# Network Parameters
img_size = 64
n_channel = 3
mask_size = 25

# pathway: eye_left and eye_right
conv1_eye_size = 11
conv1_eye_out = 96
pool1_eye_size = 2
pool1_eye_stride = 2

conv2_eye_size = 5
conv2_eye_out = 256
pool2_eye_size = 2
pool2_eye_stride = 2

conv3_eye_size = 3
conv3_eye_out = 384
pool3_eye_size = 2
pool3_eye_stride = 2

conv4_eye_size = 1
conv4_eye_out = 64
pool4_eye_size = 2
pool4_eye_stride = 2

eye_size = 2 * 2 * 2 * conv4_eye_out

# pathway: face
conv1_face_size = 11
conv1_face_out = 96
pool1_face_size = 2
pool1_face_stride = 2

conv2_face_size = 5
conv2_face_out = 256
pool2_face_size = 2
pool2_face_stride = 2

conv3_face_size = 3
conv3_face_out = 384
pool3_face_size = 2
pool3_face_stride = 2

conv4_face_size = 1
conv4_face_out = 64
pool4_face_size = 2
pool4_face_stride = 2

face_size = 2 * 2 * conv4_face_out

# fc layer
fc_eye_size = 128
fc_face_size = 128
fc2_face_size = 64
fc_face_mask_size = 256
fc2_face_mask_size = 128
fc_size = 128
fc2_size = 2


# Import data
def load_data(file):
    npzfile = np.load(file)
    train_eye_left = npzfile["train_eye_left"]
    train_eye_right = npzfile["train_eye_right"]
    train_face = npzfile["train_face"]
    train_face_mask = npzfile["train_face_mask"]
    train_y = npzfile["train_y"]
    val_eye_left = npzfile["val_eye_left"]
    val_eye_right = npzfile["val_eye_right"]
    val_face = npzfile["val_face"]
    val_face_mask = npzfile["val_face_mask"]
    val_y = npzfile["val_y"]
    return [train_eye_left, train_eye_right, train_face, train_face_mask, train_y], [val_eye_left, val_eye_right, val_face, val_face_mask, val_y]

def normalize(data):
    shape = data.shape
    data = np.reshape(data, (shape[0], -1))
    data = data.astype('float32') / 255. # scaling
    data = data - np.mean(data, axis=0) # normalizing
    return np.reshape(data, shape)

def prepare_data(data):
    eye_left, eye_right, face, face_mask, y = data
    eye_left = normalize(eye_left)
    eye_right = normalize(eye_right)
    face = normalize(face)
    face_mask = np.reshape(face_mask, (face_mask.shape[0], -1)).astype('float32')
    y = y.astype('float32')
    return [eye_left, eye_right, face, face_mask, y]

def shuffle_data(data):
    idx = np.arange(data[0].shape[0])
    np.random.shuffle(idx)
    for i in range(len(data)):
        data[i] = data[i][idx]
    return data

def next_batch(data, batch_size):
    for i in np.arange(0, data[0].shape[0], batch_size):
        # yield a tuple of the current batched data
        yield [each[i: i + batch_size] for each in data]

class EyeTracker(object):
    def __init__(self):
        # tf Graph input
        self.eye_left = tf.placeholder(tf.float32, [None, img_size, img_size, n_channel], name='eye_left')
        self.eye_right = tf.placeholder(tf.float32, [None, img_size, img_size, n_channel], name='eye_right')
        self.face = tf.placeholder(tf.float32, [None, img_size, img_size, n_channel], name='face')
        self.face_mask = tf.placeholder(tf.float32, [None, mask_size * mask_size], name='face_mask')
        self.y = tf.placeholder(tf.float32, [None, 2], name='pos')
        # Store layers weight & bias
        self.weights = {
            'conv1_eye': tf.get_variable('conv1_eye_w', shape=(conv1_eye_size, conv1_eye_size, n_channel, conv1_eye_out), initializer=tf.contrib.layers.xavier_initializer()),
            'conv2_eye': tf.get_variable('conv2_eye_w', shape=(conv2_eye_size, conv2_eye_size, conv1_eye_out, conv2_eye_out), initializer=tf.contrib.layers.xavier_initializer()),
            'conv3_eye': tf.get_variable('conv3_eye_w', shape=(conv3_eye_size, conv3_eye_size, conv2_eye_out, conv3_eye_out), initializer=tf.contrib.layers.xavier_initializer()),
            'conv4_eye': tf.get_variable('conv4_eye_w', shape=(conv4_eye_size, conv4_eye_size, conv3_eye_out, conv4_eye_out), initializer=tf.contrib.layers.xavier_initializer()),
            'conv1_face': tf.get_variable('conv1_face_w', shape=(conv1_face_size, conv1_face_size, n_channel, conv1_face_out), initializer=tf.contrib.layers.xavier_initializer()),
            'conv2_face': tf.get_variable('conv2_face_w', shape=(conv2_face_size, conv2_face_size, conv1_face_out, conv2_face_out), initializer=tf.contrib.layers.xavier_initializer()),
            'conv3_face': tf.get_variable('conv3_face_w', shape=(conv3_face_size, conv3_face_size, conv2_face_out, conv3_face_out), initializer=tf.contrib.layers.xavier_initializer()),
            'conv4_face': tf.get_variable('conv4_face_w', shape=(conv4_face_size, conv4_face_size, conv3_face_out, conv4_face_out), initializer=tf.contrib.layers.xavier_initializer()),
            'fc_eye': tf.get_variable('fc_eye_w', shape=(eye_size, fc_eye_size), initializer=tf.contrib.layers.xavier_initializer()),
            'fc_face': tf.get_variable('fc_face_w', shape=(face_size, fc_face_size), initializer=tf.contrib.layers.xavier_initializer()),
            'fc2_face': tf.get_variable('fc2_face_w', shape=(fc_face_size, fc2_face_size), initializer=tf.contrib.layers.xavier_initializer()),
            'fc_face_mask': tf.get_variable('fc_face_mask_w', shape=(mask_size * mask_size, fc_face_mask_size), initializer=tf.contrib.layers.xavier_initializer()),
            'fc2_face_mask': tf.get_variable('fc2_face_mask_w', shape=(fc_face_mask_size, fc2_face_mask_size), initializer=tf.contrib.layers.xavier_initializer()),
            'fc': tf.get_variable('fc_w', shape=(fc_eye_size + fc2_face_size + fc2_face_mask_size, fc_size), initializer=tf.contrib.layers.xavier_initializer()),
            'fc2': tf.get_variable('fc2_w', shape=(fc_size, fc2_size), initializer=tf.contrib.layers.xavier_initializer())
        }
        self.biases = {
            'conv1_eye': tf.Variable(tf.constant(0.1, shape=[conv1_eye_out])),
            'conv2_eye': tf.Variable(tf.constant(0.1, shape=[conv2_eye_out])),
            'conv3_eye': tf.Variable(tf.constant(0.1, shape=[conv3_eye_out])),
            'conv4_eye': tf.Variable(tf.constant(0.1, shape=[conv4_eye_out])),
            'conv1_face': tf.Variable(tf.constant(0.1, shape=[conv1_face_out])),
            'conv2_face': tf.Variable(tf.constant(0.1, shape=[conv2_face_out])),
            'conv3_face': tf.Variable(tf.constant(0.1, shape=[conv3_face_out])),
            'conv4_face': tf.Variable(tf.constant(0.1, shape=[conv4_face_out])),
            'fc_eye': tf.Variable(tf.constant(0.1, shape=[fc_eye_size])),
            'fc_face': tf.Variable(tf.constant(0.1, shape=[fc_face_size])),
            'fc2_face': tf.Variable(tf.constant(0.1, shape=[fc2_face_size])),
            'fc_face_mask': tf.Variable(tf.constant(0.1, shape=[fc_face_mask_size])),
            'fc2_face_mask': tf.Variable(tf.constant(0.1, shape=[fc2_face_mask_size])),
            'fc': tf.Variable(tf.constant(0.1, shape=[fc_size])),
            'fc2': tf.Variable(tf.constant(0.1, shape=[fc2_size]))
        }

        # Construct model
        self.pred = self.itracker_nets(self.eye_left, self.eye_right, self.face, self.face_mask, self.weights, self.biases)

    # Create some wrappers for simplicity
    def conv2d(self, x, W, b, strides=1):
        # Conv2D wrapper, with bias and relu activation
        x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='VALID')
        x = tf.nn.bias_add(x, b)
        return tf.nn.relu(x)

    def maxpool2d(self, x, k, strides):
        # MaxPool2D wrapper
        return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, strides, strides, 1],
                              padding='VALID')

    # Create model
    def itracker_nets(self, eye_left, eye_right, face, face_mask, weights, biases):
        # pathway: left eye
        eye_left = self.conv2d(eye_left, weights['conv1_eye'], biases['conv1_eye'], strides=1)
        eye_left = self.maxpool2d(eye_left, k=pool1_eye_size, strides=pool1_eye_stride)

        eye_left = self.conv2d(eye_left, weights['conv2_eye'], biases['conv2_eye'], strides=1)
        eye_left = self.maxpool2d(eye_left, k=pool2_eye_size, strides=pool2_eye_stride)

        eye_left = self.conv2d(eye_left, weights['conv3_eye'], biases['conv3_eye'], strides=1)
        eye_left = self.maxpool2d(eye_left, k=pool3_eye_size, strides=pool3_eye_stride)

        eye_left = self.conv2d(eye_left, weights['conv4_eye'], biases['conv4_eye'], strides=1)
        eye_left = self.maxpool2d(eye_left, k=pool4_eye_size, strides=pool4_eye_stride)

        # pathway: right eye
        eye_right = self.conv2d(eye_right, weights['conv1_eye'], biases['conv1_eye'], strides=1)
        eye_right = self.maxpool2d(eye_right, k=pool1_eye_size, strides=pool1_eye_stride)

        eye_right = self.conv2d(eye_right, weights['conv2_eye'], biases['conv2_eye'], strides=1)
        eye_right = self.maxpool2d(eye_right, k=pool2_eye_size, strides=pool2_eye_stride)

        eye_right = self.conv2d(eye_right, weights['conv3_eye'], biases['conv3_eye'], strides=1)
        eye_right = self.maxpool2d(eye_right, k=pool3_eye_size, strides=pool3_eye_stride)

        eye_right = self.conv2d(eye_right, weights['conv4_eye'], biases['conv4_eye'], strides=1)
        eye_right = self.maxpool2d(eye_right, k=pool4_eye_size, strides=pool4_eye_stride)

        # pathway: face
        face = self.conv2d(face, weights['conv1_face'], biases['conv1_face'], strides=1)
        face = self.maxpool2d(face, k=pool1_face_size, strides=pool1_face_stride)

        face = self.conv2d(face, weights['conv2_face'], biases['conv2_face'], strides=1)
        face = self.maxpool2d(face, k=pool2_face_size, strides=pool2_face_stride)

        face = self.conv2d(face, weights['conv3_face'], biases['conv3_face'], strides=1)
        face = self.maxpool2d(face, k=pool3_face_size, strides=pool3_face_stride)

        face = self.conv2d(face, weights['conv4_face'], biases['conv4_face'], strides=1)
        face = self.maxpool2d(face, k=pool4_face_size, strides=pool4_face_stride)

        # fc layer
        # eye
        eye_left = tf.reshape(eye_left, [-1, int(np.prod(eye_left.get_shape()[1:]))])
        eye_right = tf.reshape(eye_right, [-1, int(np.prod(eye_right.get_shape()[1:]))])
        eye = tf.concat([eye_left, eye_right], 1)
        eye = tf.nn.relu(tf.add(tf.matmul(eye, weights['fc_eye']), biases['fc_eye']))

        # face
        face = tf.reshape(face, [-1, int(np.prod(face.get_shape()[1:]))])
        face = tf.nn.relu(tf.add(tf.matmul(face, weights['fc_face']), biases['fc_face']))
        face = tf.nn.relu(tf.add(tf.matmul(face, weights['fc2_face']), biases['fc2_face']))

        # face mask
        face_mask = tf.nn.relu(tf.add(tf.matmul(face_mask, weights['fc_face_mask']), biases['fc_face_mask']))
        face_mask = tf.nn.relu(tf.add(tf.matmul(face_mask, weights['fc2_face_mask']), biases['fc2_face_mask']))

        # all
        fc = tf.concat([eye, face, face_mask], 1)
        fc = tf.nn.relu(tf.add(tf.matmul(fc, weights['fc']), biases['fc']))
        out = tf.add(tf.matmul(fc, weights['fc2']), biases['fc2'])
        return out

    def train(self, train_data, val_data, lr=1e-3, batch_size=128, max_epoch=1000, min_delta=1e-4, patience=10, print_per_epoch=10, out_model='my_model'):
        ckpt = os.path.split(out_model)[0]
        if not os.path.exists(ckpt):
            os.makedirs(ckpt)

        print 'Train on %s samples, validate on %s samples' % (train_data[0].shape[0], val_data[0].shape[0])
        # Define loss and optimizer
        self.cost = tf.losses.mean_squared_error(self.y, self.pred)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(self.cost)

        # Evaluate model
        self.err = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.squared_difference(self.pred, self.y), axis=1)))
        train_loss_history = []
        train_err_history = []
        val_loss_history = []
        val_err_history = []
        n_incr_error = 0  # nb. of consecutive increase in error
        best_loss = np.Inf
        n_batches = train_data[0].shape[0] / batch_size + (train_data[0].shape[0] % batch_size != 0)

        # Create the collection
        tf.get_collection("validation_nodes")
        # Add stuff to the collection.
        tf.add_to_collection("validation_nodes", self.eye_left)
        tf.add_to_collection("validation_nodes", self.eye_right)
        tf.add_to_collection("validation_nodes", self.face)
        tf.add_to_collection("validation_nodes", self.face_mask)
        tf.add_to_collection("validation_nodes", self.pred)
        saver = tf.train.Saver(max_to_keep=1)

        # Initializing the variables
        init = tf.global_variables_initializer()
        # Launch the graph
        with tf.Session() as sess:
            sess.run(init)
            # Keep training until reach max iterations
            for n_epoch in range(1, max_epoch + 1):
                n_incr_error += 1
                train_loss = 0.
                val_loss = 0.
                train_err = 0.
                val_err = 0.
                train_data = shuffle_data(train_data)
                for batch_train_data in next_batch(train_data, batch_size):
                    # Run optimization op (backprop)
                    sess.run(self.optimizer, feed_dict={self.eye_left: batch_train_data[0], \
                                self.eye_right: batch_train_data[1], self.face: batch_train_data[2], \
                                self.face_mask: batch_train_data[3], self.y: batch_train_data[4]})
                    train_batch_loss, train_batch_err = sess.run([self.cost, self.err], feed_dict={self.eye_left: batch_train_data[0], \
                                self.eye_right: batch_train_data[1], self.face: batch_train_data[2], \
                                self.face_mask: batch_train_data[3], self.y: batch_train_data[4]})
                    train_loss += train_batch_loss / n_batches
                    train_err += train_batch_err / n_batches
                val_loss, val_err = sess.run([self.cost, self.err], feed_dict={self.eye_left: val_data[0], \
                                self.eye_right: val_data[1], self.face: val_data[2], \
                                self.face_mask: val_data[3], self.y: val_data[4]})

                train_loss_history.append(train_loss)
                train_err_history.append(train_err)
                val_loss_history.append(val_loss)
                val_err_history.append(val_err)
                if val_loss - min_delta < best_loss:
                    best_loss = val_loss
                    save_path = saver.save(sess, out_model, global_step=n_epoch)
                    print "Model saved in file: %s" % save_path
                    n_incr_error = 0

                if n_epoch % print_per_epoch == 0:
                    print 'Epoch %s/%s, train loss: %.5f, train error: %.5f, val loss: %.5f, val error: %.5f' % \
                                                (n_epoch, max_epoch, train_loss, train_err, val_loss, val_err)

                if n_incr_error >= patience:
                    print 'Early stopping occured. Optimization Finished!'
                    return train_loss_history, train_err_history, val_loss_history, val_err_history

            return train_loss_history, train_err_history, val_loss_history, val_err_history

def extract_validation_handles(session):
    """ Extracts the input and predict_op handles that we use for validation.
    Args:
        session: The session with the loaded graph.
    Returns:
        validation handles.
    """
    valid_nodes = tf.get_collection_ref("validation_nodes")
    if len(valid_nodes) != 5:
        raise Exception("ERROR: Expected 5 items in validation_nodes, got %d." % len(valid_nodes))
    return valid_nodes

def load_model(session, save_path):
    """ Loads a saved TF model from a file.
    Args:
        session: The tf.Session to use.
        save_path: The save path for the saved session, returned by Saver.save().
    Returns:
        The inputs placehoder and the prediction operation.
    """
    print "Loading model from file '%s'..." % save_path

    meta_file = save_path + ".meta"
    if not os.path.exists(meta_file):
        raise Exception("ERROR: Expected .meta file '%s', but could not find it." % meta_file)

    saver = tf.train.import_meta_graph(meta_file)
    # It's finicky about the save path.
    save_path = os.path.join("./", save_path)
    saver.restore(session, save_path)

    # Check that we have the handles we expected.
    return extract_validation_handles(session)

def validate_model(session, val_data, val_ops):
    """ Validates the model stored in a session.
    Args:
        session: The session where the model is loaded.
        val_data: The validation data to use for evaluating the model.
        val_ops: The validation operations.
    Returns:
        The overall validation error for the model. """
    print "Validating model..."

    eye_left, eye_right, face, face_mask, pred = val_ops
    val_eye_left, val_eye_right, val_face, val_face_mask, val_y = val_data
    y = tf.placeholder(tf.float32, [None, 2], name='pos')
    err = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.squared_difference(pred, y), axis=1)))
    # Validate the model.
    error = session.run(err, feed_dict={eye_left: val_eye_left, \
                                eye_right: val_eye_right, face: val_face, \
                                face_mask: val_face_mask, y: val_y})
    return error

def plot_loss(train_loss, train_err, test_err, start=0, per=1, save_file='loss.png'):
    assert len(train_err) == len(test_err)
    idx = np.arange(start, len(train_loss), per)
    fig, ax1 = plt.subplots()
    lns1 = ax1.plot(idx, train_loss[idx], 'b-', alpha=1.0, label='train loss')
    ax1.set_xlabel('epochs')
    # Make the y-axis label, ticks and tick labels match the line color.
    ax1.set_ylabel('loss', color='b')
    ax1.tick_params('y', colors='b')

    ax2 = ax1.twinx()
    lns2 = ax2.plot(idx, train_err[idx], 'r-', alpha=1.0, label='train error')
    lns3 = ax2.plot(idx, test_err[idx], 'g-', alpha=1.0, label='test error')
    ax2.set_ylabel('error', color='r')
    ax2.tick_params('y', colors='r')

    # added these three lines
    lns = lns1 + lns2 + lns3
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc=0)

    fig.tight_layout()
    plt.savefig(save_file)
    # plt.show()

def train(args):
    train_data, val_data = load_data(args.input)

    # train_size = 10
    # train_data = [each[:train_size] for each in train_data]
    # val_size = 1
    # val_data = [each[:val_size] for each in val_data]
    train_data = prepare_data(train_data)
    val_data = prepare_data(val_data)

    start = timeit.default_timer()
    et = EyeTracker()
    train_loss_history, train_err_history, val_loss_history, val_err_history = et.train(train_data, val_data, \
                                            lr=args.learning_rate, \
                                            batch_size=args.batch_size, \
                                            max_epoch=args.max_epoch, \
                                            min_delta=1e-4, \
                                            patience=args.patience, \
                                            print_per_epoch=args.print_per_epoch,
                                            out_model=args.save_model)

    print 'runtime: %.1fs' % (timeit.default_timer() - start)

    if args.save_loss:
        with open(args.save_loss, 'w') as outfile:
            np.savez(outfile, train_loss_history=train_loss_history, train_err_history=train_err_history, \
                                    val_loss_history=val_loss_history, val_err_history=val_err_history)

    if args.plot_loss:
        plot_loss(np.array(train_loss_history), np.array(train_err_history), np.array(val_err_history), start=0, per=1, save_file=args.plot_loss)

def test(args):
    _, val_data = load_data(args.input)

    # val_size = 10
    # val_data = [each[:val_size] for each in val_data]

    val_data = prepare_data(val_data)

    # Load and validate the network.
    with tf.Session() as sess:
        val_ops = load_model(sess, args.load_model)
        error = validate_model(sess, val_data, val_ops)
        print 'Overall validation error: %f' % error

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true', help='train flag')
    parser.add_argument('-i', '--input', required=True, type=str, help='path to the input data')
    parser.add_argument('-max_epoch', '--max_epoch', type=int, default=100, help='max number of iterations (default 100)')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.002, help='learning rate (default 1e-3)')
    parser.add_argument('-bs', '--batch_size', type=int, default=128, help='batch size (default 50)')
    parser.add_argument('-p', '--patience', type=int, default=5, help='early stopping patience (default 10)')
    parser.add_argument('-pp_iter', '--print_per_epoch', type=int, default=1, help='print per iteration (default 10)')
    parser.add_argument('-sm', '--save_model', type=str, default='my_model', help='path to the output model (default my_model)')
    parser.add_argument('-lm', '--load_model', type=str, help='path to the loaded model')
    parser.add_argument('-pf', '--plot_filter', type=str, default='filter.png', help='plot filters')
    parser.add_argument('-pl', '--plot_loss', type=str, default='loss.png', help='plot loss')
    parser.add_argument('-sl', '--save_loss', type=str, default='loss.npz', help='save loss')
    args = parser.parse_args()

    if args.train:
        train(args)
    else:
        if not args.load_model:
            raise Exception('load_model arg needed in test phase')
        test(args)

if __name__ == '__main__':
    main()
