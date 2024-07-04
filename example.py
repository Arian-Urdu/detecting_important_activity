# coding=utf-8
# Copyright 2024 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utilities to load and process a sign language detection dataset."""
import functools
import os
from typing import Any
from typing import Dict
from typing import Tuple

from pose_format.pose import Pose
from pose_format.pose_header import PoseHeader
from pose_format.tensorflow.masked.tensor import MaskedTensor
from pose_format.tensorflow.pose_body import TensorflowPoseBody
from pose_format.tensorflow.pose_body import TF_POSE_RECORD_DESCRIPTION
from pose_format.utils.reader import BufferReader
import tensorflow as tf

from args import FLAGS

# from pose_format.utils.openpose import load_openpose_json



@functools.lru_cache(maxsize=1)
def get_openpose_header():
  """Get pose header with OpenPose components description."""
  dir_path = os.path.dirname(os.path.realpath(__file__))
  header_path = os.path.join(dir_path, "assets/openpose.poseheader")
  f = open(header_path, "rb")
  reader = BufferReader(f.read())
  header = PoseHeader.read(reader)
  return header


def differentiate_frames(src):
  """Subtract every two consecutive frames."""
  # Shift data to pre/post frames
  pre_src = src[:-1]
  post_src = src[1:]

  # Differentiate src points
  src = pre_src - post_src

  return src


def distance(src):
  """Calculate the Euclidean distance from x:y coordinates."""
  square = src.square()
  sum_squares = square.sum(dim=-1).fix_nan()
  sqrt = sum_squares.sqrt().zero_filled()
  return sqrt


def optical_flow(src, fps):
  """Calculate the optical flow norm between frames, normalized by fps."""

  # Remove "people" dimension
  src = src.squeeze(1)

  print(src)
  # Differentiate Frames
  src = differentiate_frames(src)

  # Calculate distance
  src = distance(src)

  # Normalize distance by fps
  src = src * fps

  return src


minimum_fps = tf.constant(1, dtype=tf.float32)

def load_datum(tfrecord_dict):
  """Convert tfrecord dictionary to tensors."""
  pose_body = TensorflowPoseBody.from_tfrecord(tfrecord_dict)
 
  pose = Pose(header=get_openpose_header(), body=pose_body)
  #tgt = tf.io.decode_raw(tfrecord_dict["is_signing"], out_type=tf.int8)

  #fps = pose.body.fps
  #frames = tf.cast(tf.size(tgt), dtype=fps.dtype)

  # Get only relevant input components
  #pose = pose.get_components(FLAGS.input_components)
  datum = {
      "fps": pose.body.fps,
      "pose_data_tensor": pose.body.data.tensor,
      "pose_data_mask": pose.body.data.mask,
      "pose_confidence": pose.body.confidence,
  }

  print(datum)
  return datum
def get_input(datum,
                  augment=False):
  
  masked_tensor = MaskedTensor(
      tensor=datum["pose_data_tensor"], mask=datum["pose_data_mask"])
  pose_body = TensorflowPoseBody(
      fps=datum["fps"], data=masked_tensor, confidence=datum["pose_confidence"])
  pose = Pose(header=get_openpose_header(), body=pose_body)

  fps = pose.body.fps

  if augment:
    pose, selected_indexes = pose.frame_dropout(FLAGS.frame_dropout_std)
    tgt = tf.gather(tgt, selected_indexes)

    new_frames = tf.cast(tf.size(tgt), dtype=fps.dtype)

    fps = tf.math.maximum(minimum_fps, (new_frames / frames) * fps)
    frames = new_frames

  flow = optical_flow(pose.body.data, fps)

 # Debugging: Check shapes of tensors
  print(flow)
  return {"src": flow}

# Define the feature description
feature_description = {
    'fps': tf.io.FixedLenFeature([], tf.int64),
    'pose_data': tf.io.FixedLenFeature([], tf.string),
    'pose_confidence': tf.io.FixedLenFeature([], tf.string),
}

# Create a parsing function
def _parse_function(example_proto):
    # Parse the input tf.train.Example proto using the feature description dictionary
    return tf.io.parse_single_example(example_proto, feature_description)

# Function to convert a TFRecord file to a dictionary of lists
def tfrecord_to_dict(tfrecord_file):
    # Create a dataset from the TFRecord file
    raw_dataset = tf.data.TFRecordDataset(tfrecord_file)
    
    # Parse the dataset
    parsed_dataset = raw_dataset.map(_parse_function)

    # Initialize the dictionary to store lists of feature values
    records_dict = {
        'fps': [],
        'pose_data': [],
        'pose_confidence': []
    }

    # Convert the parsed dataset to a dictionary of lists
    for parsed_record in parsed_dataset:
        records_dict['fps'].append(parsed_record['fps'].numpy())
        records_dict['pose_data'].append(tf.io.parse_tensor(parsed_record['pose_data'], out_type=tf.string).numpy())
        records_dict['pose_confidence'].append(tf.io.parse_tensor(parsed_record['pose_confidence'], out_type=tf.float32).numpy())
        
    return records_dict

def recover(_, y):
    return y

def prepare_io(datum):
  """Convert dictionary into input-output tuple for Keras."""
  src = datum["src"]
  print(src)
  return src


def batch_dataset(dataset, batch_size):
  """Batch and pad a dataset."""
  dataset = dataset.padded_batch(
      batch_size, padded_shapes={
          "src": [None, None]
      })

  return dataset.map(prepare_io)

BATCH_SIZE = 1
AUTOTUNE = tf.data.experimental.AUTOTUNE


# Path to the TFRecord file
tfrecord_file = 'sample.tfrecord'

# Create a dataset from the TFRecord file
raw_dataset = tf.data.TFRecordDataset(tfrecord_file)

# Parse and process the dataset
dataset = raw_dataset.map(_parse_function)
dataset = dataset.map(load_datum)
dataset = dataset.map(get_input)

# Batch and prefetch the dataset
dataset = batch_dataset(dataset, BATCH_SIZE)


# Load the trained model
model_path = 'models/py/model.h5'
model = tf.keras.models.load_model(model_path)

# Display model summary
model.summary()

# For demonstration purposes, we'll take one batch from the dataset
for batch in dataset.take(1):
    #print(f"using: {batch}")
    print("Running Model...")
    predictions = model(batch)
    print(predictions)