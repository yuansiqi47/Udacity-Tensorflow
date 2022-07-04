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
parser.add_argument(action='store', dest='image_path',
                    default='./test_images/cautleya_spicata.jpg')

parser.add_argument(action='store', dest='model_dir',
                    default='my_model.h5')

parser.add_argument('--top_k', action='store', dest='top_k',
                    default=1)

parser.add_argument('--category_names', action='store',
                    dest='category_names', default='label_map.json')

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
with open(args.category_names, 'r') as f:
    class_names = json.load(f)

# Get the probability and the predicted class
for i in range(int(args.top_k)):
    prob = max(probs)
    index = probs.index(prob) + 1
    cls = class_names[str(index)]
    # Print the probability and the predicted class
    print(prob)
    print(cls)
    probs.remove(prob)
    i += 1
