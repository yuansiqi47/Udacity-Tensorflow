import numpy as np
import tensorflow as tf
from PIL import Image
import json
import argparse
import tensorflow as tf
import tensorflow_hub as hub

batch_size = 64

# setting
parser = argparse.ArgumentParser()
parser.add_argument('--image_path', action='store', dest='image_path',
                    default='./test_images/cautleya_spicata.jpg')

parser.add_argument('--model_dir', action='store',
                    dest='model_dir', default='my_model.h5')

args = parser.parse_args()

# load the model
save_dir = args.model_dir
model = tf.keras.experimental.load_from_saved_model(
    save_dir, custom_objects={'KerasLayer': hub.KerasLayer})

# load the image
image = Image.open(args.image_path)

# process the image
image = np.array(image)
image = tf.cast(image, tf.float32)
image = tf.image.resize(image, (224, 224))
image /= 255
image = np.expand_dims(image, axis=0)

# make prediction
probs = model.predict(image).flatten().tolist()

# open class_names file
with open('label_map.json', 'r') as f:
    class_names = json.load(f)

# Get the probability and the predicted class
prob = max(probs)
index = probs.index(prob) + 1
cls = class_names[str(index)]

# Print the probability and the predicted class
print(prob)
print(cls)