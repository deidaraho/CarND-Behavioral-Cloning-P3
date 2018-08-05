import cv2
import csv
import numpy as np
import os
from keras.models import load_model

def get_lines_from_log(path, skip_header=False):
    lines = []
    with open(path+'/driving_log.csv') as file:
        reader = csv.reader(file)
        if skip_header:
            next(reader, None)
        for line in reader:
            lines.append(line)
    return lines

def find_images(path):
    directories = [x[0] for x in os.walk(path)]
    data_directories = list(filter(lambda directory: os.path.isfile(directory + '/driving_log.csv'), directories))
    center_total = []
    left_total = []
    right_total = []
    measurement_total = []
    for directory in data_directories:
        lines = get_lines_from_log(directory)
        center = []
        left = []
        right = []
        measurements = []
        for line in lines:
            measurements.append(float(line[3]))
            center.append(line[0].strip())
            left.append(line[1].strip())
            right.append(line[2].strip())
        center_total.extend(center)
        left_total.extend(left)
        right_total.extend(right)
        measurement_total.extend(measurements)

    return (center_total, left_total, right_total, measurement_total)

def combine_images(center, left, right, measurement, correction):
    image_paths = []
    image_paths.extend(center)
    image_paths.extend(left)
    image_paths.extend(right)
    measurements = []
    measurements.extend(measurement)
    measurements.extend([x + correction for x in measurement])
    measurements.extend([x - correction for x in measurement])
    return (image_paths, measurements)

import sklearn

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        samples = sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset: offset+batch_size]
            images = []
            angles = []

            for image_path, measurement in batch_samples:
                original_image = cv2.imread(image_path)
                image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
                images.append(image)
                angles.append(measurement)
                # Flippingc
                images.append(cv2.flip(image,1))
                angles.append(measurement*-1.0)

            # trim image to only see section with road
            inputs = np.array(images)
            outputs = np.array(angles)
            yield sklearn.utils.shuffle(inputs, outputs)

def test_samples(samples):
    for image_path, number in samples:
        try:
            original_image = cv2.imread(image_path)
            image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        except:
            print ("cannot open: " + image_path)

from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Lambda, Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D
import matplotlib.pyplot as plt

def create_pre_processing_layers():
    model = Sequential()
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160, 320, 3)))
    model.add(Cropping2D(cropping=((50,20), (0,0))))
    return model

def nVidiaModel():
    model = create_pre_processing_layers()
    model.add(Convolution2D(24,5,5, subsample=(2,2), activation='relu'))
    model.add(Convolution2D(36,5,5, subsample=(2,2), activation='relu'))
    model.add(Convolution2D(48,5,5, subsample=(2,2), activation='relu'))
    model.add(Convolution2D(64,3,3, activation='relu'))
    model.add(Convolution2D(64,3,3, activation='relu'))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))
    return model

if __name__ == '__main__':
    # Reading images locations.
    center_paths, left_paths, right_paths, measurements = find_images('./data/')
    image_paths, measurements = combine_images(center_paths, left_paths, right_paths, measurements, 0.2)
    print('Total Images: {}'.format(len(image_paths)))

    # Splitting samples and creating generators.
    from sklearn.model_selection import train_test_split
    samples = list(zip(image_paths, measurements))
    train_samples, validation_samples = train_test_split(samples, test_size=0.2)

    print('Train samples: {}'.format(len(train_samples)))
    print('Validation samples: {}'.format(len(validation_samples)))

    #test_samples(train_samples)
    #test_samples(validation_samples)
    #import pdb; pdb.set_trace()

    train_generator = generator(train_samples, batch_size=32)
    validation_generator = generator(validation_samples, batch_size=32)

    # Model creation
    try:
        model = load_model('./model.h5')
        print('loading previouse model and tuning.')
    except:
        print('No previouse model for tuning.')
        model = nVidiaModel()

    # Compiling and training the model
    model.compile(loss='mse', optimizer='adam')
    history_object = model.fit_generator(train_generator,
    samples_per_epoch=len(train_samples),
    validation_data=validation_generator,
    nb_val_samples=len(validation_samples),
    nb_epoch=3, verbose=1)

    model.save('model.h5')
    print(history_object.history.keys())
    print('Loss')
    print(history_object.history['loss'])
    print('Validation Loss')
    print(history_object.history['val_loss'])
