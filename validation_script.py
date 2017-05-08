#!/usr/bin/python

import argparse
"""
try:
  import cPickle as pickle
except ImportError:
  # Python 3
  import pickle
"""
import os
import sys

import numpy as np

import tensorflow as tf


# How many images to include in each validation batch. This is just a default
# value, and may be set differently to accomodate network parameters.
batch_size = 1000


def extract_validation_handles(session):
  """ Extracts the input and predict_op handles that we use for validation.
  Args:
    session: The session with the loaded graph.
  Returns:
    The inputs placeholder, mask placeholder, and the prediction operation. """
  # The students should have saved their input placeholder, mask placeholder and prediction
  # operation in a collection called "validation_nodes".
  valid_nodes = tf.get_collection_ref("validation_nodes")
  if len(valid_nodes) != 5:
    print("ERROR: Expected 3 items in validation_nodes, got %d." % \
          (len(valid_nodes)))
    sys.exit(1)

  # Figure out which is which.
  eye_left = valid_nodes[0]
  eye_right = valid_nodes[1]
  face = valid_nodes[2]
  face_mask = valid_nodes[3]
  predict = valid_nodes[4]
  """if type(valid_nodes[1]) == tf.placeholder:
    inputs = valid_nodes[1]
    predict = valid_nodes[0]"""

  # Check to make sure we've set the batch size correctly.
  global batch_size
  try:
    batch_size = int(eye_left.get_shape()[0])
    print("WARNING: Network does not support variable batch sizes. (inputs)")
  except TypeError:
    # It's unspecified, which is actually correct.
    pass
  try:
    # I've also seen people who don't specify an input shape but do specify a
    # shape for the prediction operation.
    batch_size = int(predict.get_shape()[0])
    print("WARNING: Network does not support variable batch sizes. (predict)")
  except TypeError:
    pass

  # Predict op should also yield integers.
  #predict = tf.cast(predict, "int32")

  # Check the shape of the prediction output.
  p_shape = predict.get_shape()
  #Commented these out because there could be squeezes in the code earlier
  """
  print p_shape
  if len(p_shape) > 2:
    print("ERROR: Expected prediction of shape (<X>, 1), got shape of %s." % \
          (str(p_shape)))
    sys.exit(1)
  if len(p_shape) == 2:
    if p_shape[1] != 1:
      print("ERROR: Expected prediction of shape (<X>, 1), got shape of %s." % \
            (str(p_shape)))
      sys.exit(1)

    # We need to contract it into a vector.
    predict = predict[:, 0]"""

  return (eye_left, eye_right, face, face_mask, predict)

def load_model(session, save_path):
  """ Loads a saved TF model from a file.
  Args:
    session: The tf.Session to use.
    save_path: The save path for the saved session, returned by Saver.save().
  Returns:
    The inputs placehoder and the prediction operation.
  """
  print("Loading model from file '%s'..." % (save_path))

  meta_file = save_path + ".meta"
  if not os.path.exists(meta_file):
    print("ERROR: Expected .meta file '%s', but could not find it." % \
          (meta_file))
    sys.exit(1)

  saver = tf.train.import_meta_graph(meta_file)
  # It's finicky about the save path.
  save_path = os.path.join("./", save_path)
  saver.restore(session, save_path)

  # Check that we have the handles we expected.
  return extract_validation_handles(session)

def load_validation_data(val_filename):
  """ Loads the validation data.
  Args:
    val_filename: The file where the validation data is stored.
  Returns:
    A tuple of the loaded validation data and validation labels. """
  print("Loading validation data...")

  npzfile = np.load(val_filename)
  val_eye_left = npzfile["val_eye_left"]
  val_eye_right = npzfile["val_eye_right"]
  val_face = npzfile["val_face"]
  val_face_mask = npzfile["val_face_mask"]
  val_y = npzfile["val_y"]

  return (val_eye_left, val_eye_right, val_face,  val_face_mask, val_y)

def validate_model(session, val_data, eye_left, eye_right, face, face_mask, predict_op):
  """ Validates the model stored in a session.
  Args:
    session: The session where the model is loaded.
    val_data: The validation data to use for evaluating the model.
    eye_left: The inputs placeholder.
    eye_right: The inputs placeholder.
    face: The inputs placeholder.
    face_mask: The inputs placeholder.
    predict_op: The prediction operation.
  Returns:
    The overall validation accuracy for the model. """
  print("Validating model...")



  # Validate the model.
  val_eye_left, val_eye_right, val_face,  val_face_mask, val_y = val_data
  num_iters = val_eye_left.shape[0] // batch_size

  err_val = []
  for i in range(0, int(num_iters)):
    start_index = i * batch_size
    end_index = start_index + batch_size

    eye_left_batch = val_eye_left[start_index:end_index, :]
    eye_right_batch = val_eye_right[start_index:end_index, :]
    face_batch = val_face[start_index:end_index, :]
    # face_mask_batch = val_face_mask[start_index:end_index, :]
    face_mask_batch = np.reshape(val_face_mask[start_index:end_index, :], (batch_size, -1))
    y_batch = val_y[start_index:end_index, :]



    print("Validating batch %d of %d..." % (i + 1, num_iters))
    yp = session.run(predict_op,
                     feed_dict={eye_left: eye_left_batch / 255.,
                                eye_right: eye_right_batch / 255.,
                                face: face_batch / 255.,
                                face_mask: face_mask_batch})

    err = np.mean(np.sqrt(np.sum((yp - y_batch)**2, axis=1)))
    err_val.append(err)

  # Compute total error
  error = np.mean(err_val)
  return error

def try_with_random_data(session, eye_left, eye_right, face, face_mask, predict_op):
  """ Tries putting random data through the network, mostly to make sure this
  works.
  Args:
    session: The session to use.
    inputs: The inputs placeholder.
    predict_op: The prediction operation. """
  print("Trying random batch...")

  # Get a random batch.
  eye_left_batch = np.random.rand(batch_size, 64, 64, 3)
  eye_right_batch = np.random.rand(batch_size, 64, 64, 3)
  face_batch = np.random.rand(batch_size, 64, 64, 3)
  face_mask_batch = np.random.rand(batch_size, 25, 25)

  print("Batch of shape (%d, 64, 64, 3)" % (batch_size))

  # Put it through the model.
  predictions = session.run(predict_op, feed_dict={eye_left: eye_left_batch,
                                                   eye_right: eye_right_batch,
                                                   face: face_batch,
                                                   face_mask: face_mask_batch})
  if np.isnan(predictions).any():
    print("Warning: Got NaN value in prediction!")


def main():
  parser = argparse.ArgumentParser("Analyze student models.")
  parser.add_argument("-v", "--val_data_file", default=None,
                      help="Validate the network with the data from this " + \
                           "pickle file.")
  parser.add_argument("save_path", help="The base path for your saved model.")
  args = parser.parse_args()

  if not args.val_data_file:
    print("Not validating, but checking network compatibility...")
  elif not os.path.exists(args.val_data_file):
    print("ERROR: Could not find validation data '%s'." % (args.val_data))
    sys.exit(1)

  # Load and validate the network.
  with tf.Session() as session:
    eye_left, eye_right, face, face_mask, predict_op = load_model(session, args.save_path)
    if args.val_data_file:
      val_data = load_validation_data(args.val_data_file)
      accuracy = validate_model(session, val_data, eye_left, eye_right, face, face_mask, predict_op)

      print("Overall validation error: %f cm" % (accuracy))
      print("Network seems good. Go ahead and submit")

    else:
      try_with_random_data(session, eye_left, eye_right, face, face_mask, predict_op)
      print("Network seems good. Go ahead and submit.")

if __name__ == "__main__":
  main()
