from __future__ import division
from __future__ import print_function

import time
import tensorflow as tf

from gcn.utils import *
from gcn.models import GCN, MLP

# Set random seed
seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)

# region Tf Flags
'''
The tf.app.flags module is a functionality provided by Tensorflow to implement command line flags for your Tensorflow 
program.If you pass these parameters by argument line then these values wont be valid.

'''
# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', 'cora', 'Dataset string.')  # 'cora', 'citeseer', 'pubmed'   #   Dataset string
flags.DEFINE_string('model', 'gcn', 'Model string.')  # 'gcn', 'gcn_cheby', 'dense'         #   Model string
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')                         #   Initial learning rate
flags.DEFINE_integer('epochs', 200, 'Number of epochs to train.')                           #   No of epochs to train
flags.DEFINE_integer('hidden1', 16, 'Number of units in hidden layer 1.')                   #   No of units in hidden layer 1
flags.DEFINE_float('dropout', 0.5, 'Dropout rate (1 - keep probability).')                  #   Dropout rate
flags.DEFINE_float('weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix.')         #   Weight for L2 loss on embedding matrix
flags.DEFINE_integer('early_stopping', 10, 'Tolerance for early stopping (# of epochs).')   #   Tolerance for early stopping (# of epochs)
flags.DEFINE_integer('max_degree', 3, 'Maximum Chebyshev polynomial degree.')               #   Maximum Chebyshev polynomial degree
# endregion

# region Dataset Load
# Load data
'''
load_data() function of util.py
'''
adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data(FLAGS.dataset)
# endregion

# region feature and Adj Matrix preprocessing step.
# Some preprocessing
features = preprocess_features(features)
if FLAGS.model == 'gcn':
    support = [preprocess_adj(adj)]                 #Normalise Adjacency Matrix
    num_supports = 1
    model_func = GCN                                # Create GCN class object model_fun
elif FLAGS.model == 'gcn_cheby':
    support = chebyshev_polynomials(adj, FLAGS.max_degree)
    num_supports = 1 + FLAGS.max_degree
    model_func = GCN
elif FLAGS.model == 'dense':
    support = [preprocess_adj(adj)]  # Not used
    num_supports = 1
    model_func = MLP
else:
    raise ValueError('Invalid argument for model: ' + str(FLAGS.model))
# endregion

# Define placeholders
placeholders = {
    'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
    'features': tf.sparse_placeholder(tf.float32, shape=tf.constant(features[2], dtype=tf.int64)),
    'labels': tf.placeholder(tf.float32, shape=(None, y_train.shape[1])),
    'labels_mask': tf.placeholder(tf.int32),
    'dropout': tf.placeholder_with_default(0., shape=()),
    'num_features_nonzero': tf.placeholder(tf.int32)  # helper variable for sparse dropout
}

# Create model
model = model_func(placeholders, input_dim=features[2][1], logging=True)

# Initialize session
sess = tf.Session()


# Define model evaluation function
def evaluate(features, support, labels, mask, placeholders):
    '''
    This procedure will evaluate accuracy for validation and test data set,
    :return: outs_val[0]:Loss   outs_val[1]:Accuracy    (time.time() - t_test):duration
    '''
    t_test = time.time()
    feed_dict_val = construct_feed_dict(features, support, labels, mask, placeholders)
    outs_val = sess.run([model.loss, model.accuracy], feed_dict=feed_dict_val)
    return outs_val[0], outs_val[1], (time.time() - t_test)


# Init variables
sess.run(tf.global_variables_initializer())

cost_val = []

# Train model
for epoch in range(FLAGS.epochs):
    t = time.time()
    # region Construct Feed Dictionary
    # Construct feed dictionary.construct_feed_dict() in utils.py is called.
    feed_dict = construct_feed_dict(features, support, y_train, train_mask, placeholders)
    #Update dropout value
    feed_dict.update({placeholders['dropout']: FLAGS.dropout})
    # endregion

    # region Training step
    outs = sess.run([model.opt_op, model.loss, model.accuracy], feed_dict=feed_dict)
    # endregion

    # region Validation Step
    cost, acc, duration = evaluate(features, support, y_val, val_mask, placeholders)
    cost_val.append(cost)
    # endregion

    # region Print Traing and Validation results
    print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(outs[1]),
          "train_acc=", "{:.5f}".format(outs[2]), "val_loss=", "{:.5f}".format(cost),
          "val_acc=", "{:.5f}".format(acc), "time=", "{:.5f}".format(time.time() - t))
    # endregion

    if epoch > FLAGS.early_stopping and cost_val[-1] > np.mean(cost_val[-(FLAGS.early_stopping+1):-1]):
        print("Early stopping...")
        break

print("Optimization Finished!")

# region Testing Results
test_cost, test_acc, test_duration = evaluate(features, support, y_test, test_mask, placeholders)
print("Test set results:", "cost=", "{:.5f}".format(test_cost),
      "accuracy=", "{:.5f}".format(test_acc), "time=", "{:.5f}".format(test_duration))
# endregion
