from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
from utils import processing_dataset
from utils import increment_path
from pathlib import Path
import argparse

def train(opt):
  data, epochs, batch_size = opt.data, opt.epochs, opt.batch_size
  save_dir, imgsz, freeze = opt.save_dir, opt.imgsz, opt.freeze
  INIT_LR = 1e-4
  Path(save_dir).mkdir(parents=True, exist_ok=False)
  wdir = Path(save_dir) / 'weights'
  wdir.mkdir(parents=True, exist_ok=True)
  trainX, trainY, testX, testY = processing_dataset(data, imgsz)
  aug = ImageDataGenerator(rotation_range = 20,
                         zoom_range = 0.15,
                         width_shift_range = 0.2,
                         height_shift_range = 0.2,
                         shear_range = 0.15,
                         horizontal_flip = True,
                         fill_mode = "nearest")
  
  baseModel = MobileNetV2(weights = "imagenet", include_top = False,
                        input_tensor = Input(shape = (imgsz, imgsz, 3)))
  headModel = baseModel.output
  headModel = AveragePooling2D(pool_size = (7, 7))(headModel)
  headModel = Flatten(name = "flatten")(headModel)
  headModel = Dense(128, activation = "relu")(headModel)
  headModel = Dropout(0.5)(headModel)
  headModel = Dense(2, activation = "softmax")(headModel)
  model = Model(inputs = baseModel.input, outputs = headModel)
  if freeze:
    for layer in baseModel.layers:
      layer.trainable = False
  
  lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=INIT_LR,
    decay_steps=10000,
    decay_rate=INIT_LR / epochs)
  opt = Adam(learning_rate = lr_schedule)
  model.compile(loss = "binary_crossentropy", optimizer = opt, metrics = ["accuracy"])
  
  file_save_model = save_dir + '/' + 'weights' + '/' + "best.h5"
  log_dir = save_dir + '/' + 'logs' 
  checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=file_save_model, 
                                                         monitor='val_accuracy', 
                                                         save_best_only=True, 
                                                         save_weights_only=False, 
                                                         mode='max', 
                                                         verbose=1)
  tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, 
                                                       histogram_freq=1, 
                                                       write_graph=True,
                                                       write_images=True) 
  print(f'Logging results to {save_dir}\n'
        f'Starting training for {epochs} epochs...')
  history = model.fit(aug.flow(trainX, trainY, batch_size = batch_size),
                    steps_per_epoch = len(trainX) // batch_size,
                    validation_data = (testX, testY),
                    validation_steps = len(testX) // batch_size,
                    epochs = epochs,
                    callbacks=[checkpoint_callback, tensorboard_callback])

  plt.style.use("ggplot")
  plt.figure()
  plt.plot(np.arange(0, epochs), history.history["loss"], label = "train_loss")
  plt.plot(np.arange(0, epochs), history.history["val_loss"], label = "val_loss")
  plt.plot(np.arange(0, epochs), history.history["accuracy"], label = "train_accuracy")
  plt.plot(np.arange(0, epochs), history.history["val_accuracy"], label = "val_accuracy")
  plt.title("Training Loss and Accuracy")
  plt.xlabel("Epochs")
  plt.ylabel("Loss/Accuracy")
  plt.legend(loc = "lower left")
  plt.savefig(Path(save_dir)/"loss_accuracy_curve.png")
if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--data', type=str, default='dataset/', help='data.yaml path')
  parser.add_argument('--epochs', type=int, default=100)
  parser.add_argument('--batch_size', type=int, default=32, help='total batch size for all GPUs')
  parser.add_argument('--imgsz', type=int, default=224, help='Size of image in training')
  parser.add_argument('--freeze', action='store_true', help='Freeze some first layer of pretrain')
  parser.add_argument('--project', default='runs/train', help='save to project/name')
  parser.add_argument('--name', default='exp', help='save to project/name')
  opt = parser.parse_args()
  opt.save_dir = increment_path(Path(opt.project) / opt.name, exist_ok=False)
  train(opt)