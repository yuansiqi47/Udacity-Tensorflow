import numpy as np
import tensorflow as tf
from PIL import Image
import json
import argparse
import tensorflow as tf
import tensorflow_hub as hub

batch_size = 64

parser = argparse.ArgumentParser()
parser.add_argument('--image_path', action='store',
                    default='./test_images/cautleya_spicata.jpg',
                    help='Enter path to image.')

parser.add_argument('--save_dir', action='store',
                    dest='save_directory', default='my_model.h5',
                    help='Enter location to save checkpoint in.')

args = parser.parse_args()
with open('label_map.json', 'r') as f:
    class_names = json.load(f)


def process_image(image):
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, (224, 224))
    image /= 255
    image = image.numpy()

    return image


def predict(image_path, model):
    image = Image.open(image_path)
    image = np.array(image)
    image = np.expand_dims(image, axis=0)
    image = process_image(image)
    probs = model.predict(image).flatten().tolist()

    prob = max(probs)
    index = probs.index(prob) + 1
    cls = class_names[str(index)]

    return prob, cls


save_dir = args.save_directory
model = tf.keras.experimental.load_from_saved_model(
    save_dir, custom_objects={'KerasLayer': hub.KerasLayer})

# Carry out prediction
prob, cls = predict(args.image_path, model)

# Print probabilities and predicted classes
print(prob)
print(cls)