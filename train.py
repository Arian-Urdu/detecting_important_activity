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
import wandb

from args import FLAGS
from dataset import get_datasets
from transformer_model import build_transformer_model

class WandbStepLogger(tf.keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.step = 0

    def on_train_batch_end(self, batch, logs=None):
        self.step += 1
        wandb.log({"train_loss": logs["loss"], "train_accuracy": logs["accuracy"]}, step=self.step)

    def on_test_batch_end(self, batch, logs=None):
        wandb.log({"val_loss": logs["loss"]}, step=self.step)

def set_seed():
    """Set seed for deterministic random number generation."""
    seed = FLAGS.seed if FLAGS.seed is not None else random.randint(0, 1000)
    tf.random.set_seed(seed)
    random.seed(seed)

def main(unused_argv):
    """Keras training loop with early-stopping, model checkpoint, and wandb tracking."""

    # Initialize wandb
    wandb.init(project="sign_language_detection", config=FLAGS)

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
    model = build_transformer_model(input_shape)

    # Callbacks
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
    
    # Add custom WandbStepLogger
    wandb_step_logger = WandbStepLogger()

    with tf.device(FLAGS.device):
        history = model.fit(
            train,
            epochs=FLAGS.epochs,
            steps_per_epoch=FLAGS.steps_per_epoch,
            validation_data=dev,
            callbacks=[es, mc, wandb_step_logger])

    best_model = tf.keras.models.load_model(FLAGS.model_path)
    print('Testing')
    test_results = best_model.evaluate(test)
    
    # Log test results to wandb
    wandb.log({
        "test_loss": test_results[0],
    })

    # Finish wandb run
    wandb.finish()

if __name__ == '__main__':
    app.run(main)