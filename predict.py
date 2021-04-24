"""
 python file
 Get Model prediction
"""

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image
import argparse
import json

batch_size = 32
image_size = 224
train_split = 60

classes = {}


def process_image(image):
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, (image_size, image_size))
    image /= 255

    return image.numpy()


def predict(img_path, f_model, f_top_k=5):
    img = Image.open(img_path)
    test_image = np.asarray(img)
    processed_test_image = process_image(test_image)
    final_image = np.expand_dims(processed_test_image, axis=0)

    prediction = f_model.predict(final_image)
    prob = - np.partition(-prediction[0], f_top_k)[:f_top_k]
    class_pred = np.argpartition(-prediction[0], f_top_k)[:f_top_k]

    return prob, class_pred


if __name__ == '__main__':
    print('predict.py, running')

    parser = argparse.ArgumentParser()
    parser.add_argument('arg1')
    parser.add_argument('arg2')
    parser.add_argument('--top_k', type=int)
    parser.add_argument('--category_names')

    args = parser.parse_args()
    print(args)

    print('arg1:', args.arg1)
    print('arg2:', args.arg2)
    print('top_k:', args.top_k)
    print('category_names:', args.category_names)

    image_path = args.arg1

    model = tf.keras.models.load_model(args.arg2, custom_objects={'KerasLayer': hub.KerasLayer})
    top_k = args.top_k
    if top_k is None:
        top_k = 5

    probs, classes = predict(image_path, model, top_k)

    if args.category_names is not None:
        with open(args.category_names, 'r') as f:
            class_names = json.load(f)
        keys = [str(x + 1) for x in list(classes)]
        classes = [class_names.get(key) for key in keys]

    print('======================================================================')
    print('Probabilities:    ', probs)
    print('Classes:          ', classes)
    print('======================================================================')
