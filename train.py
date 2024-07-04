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

"""Training script for sign language detection."""

import random

from absl import app
import tensorflow as tf
# from tf.keras.callbacks import EarlyStopping
# from tf.keras.callbacks import ModelCheckpoint
# from tf.keras.models import load_model

from args import FLAGS
from dataset import get_datasets
from model import build_model
from transformer_model import build_transformer_model


def set_seed():
  """Set seed for deterministic random number generation."""
  seed = FLAGS.seed if FLAGS.seed is not None else random.randint(0, 1000)
  tf.random.set_seed(seed)
  random.seed(seed)


def main(unused_argv):
  """Keras training loop with early-stopping and model checkpoint."""

  set_seed()

  # Initialize Dataset
  train, dev, test = get_datasets()

 # Print shapes for debugging
  for src, tgt in train.take(1):
    print("Source shape:", src.shape)
    print("Target shape:", tgt.shape)
    input_shape = (src.shape[1], src.shape[2])  # (sequence_length, feature_dim)
    break

  # Initialize Model
  #model = build_model()
  #model = build_transformer_model()


  # Adjust model input shape if necessary
  input_shape = (None, src.shape[-1])  # Dynamic sequence length, feature dimension
  model = build_transformer_model(input_shape)

  # Train
  es = tf.keras.callbacks.EarlyStopping(
      monitor='val_accuracy',
      mode='max',
      verbose=1,
      patience=FLAGS.stop_patience)
  mc = tf.keras.callbacks.ModelCheckpoint(
      FLAGS.model_path,
      monitor='val_accuracy',
      mode='max',
      verbose=1,
      save_best_only=True)

  with tf.device(FLAGS.device):
    model.fit(
        train,
        epochs=FLAGS.epochs,
        steps_per_epoch=FLAGS.steps_per_epoch,
        validation_data=dev,
        callbacks=[es, mc])

  best_model = tf.keras.models.load_model(FLAGS.model_path)
  print('Testing')
  best_model.evaluate(test)


if __name__ == '__main__':
  app.run(main)
