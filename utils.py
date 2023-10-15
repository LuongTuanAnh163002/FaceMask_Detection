from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import os
from pathlib import Path
import glob
import re
import numpy as np
from tqdm import tqdm



def increment_path(path, exist_ok=True, sep=''):
    # Increment path, i.e. runs/exp --> runs/exp{sep}0, runs/exp{sep}1 etc.
    path = Path(path)  # os-agnostic
    if (path.exists() and exist_ok) or (not path.exists()):
        return str(path)
    else:
        dirs = glob.glob(f"{path}{sep}*")  # similar paths
        matches = [re.search(rf"%s{sep}(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]  # indices
        n = max(i) + 1 if i else 2  # increment number
        return f"{path}{sep}{n}"  # update path


def processing_dataset(path, img_size):
  CATEGORIES = os.listdir(path)
  data = []
  labels = []
  print("Loading dataset....")
  for category in CATEGORIES:
    sub_path = os.path.join(path, category)
    for img in tqdm(os.listdir(sub_path)):
      img_path = os.path.join(sub_path, img)
      image = load_img(img_path, target_size = (img_size, img_size))
      image = img_to_array(image)
      image = preprocess_input(image)
      data.append(image)
      labels.append(category)
  
  lb = LabelBinarizer()
  labels = lb.fit_transform(labels)
  labels = to_categorical(labels)
  data = np.array(data, dtype = "float32")
  labels = np.array(labels)
  trainX, testX, trainY, testY = train_test_split(data, labels, test_size = 0.2, stratify = labels, random_state = 42)
  return trainX, trainY, testX, testY