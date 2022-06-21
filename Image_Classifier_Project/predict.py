import numpy as np
import tensorflow as tf
from PIL import Image
import json
import argparse

batch_size = 64

parser = argparse.ArgumentParser()
args = parser.parse_args()
with open(args.category_names, 'r') as f:
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
    probs = model.predict(image)

    prob = max(probs)
    index = probs.index(prob)
    cls = class_names[str(index)]

    return prob, cls


# Carry out prediction
prob, cls = predict(image_path, model)

# Print probabilities and predicted classes
print(prob)
print(cls)
