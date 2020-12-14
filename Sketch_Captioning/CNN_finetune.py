from keras.models import load_model
import cv2
import numpy as np
from tensorflow.keras.applications.inception_v3 import InceptionV3
import os.path
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers
from tensorflow.keras.models import Model, Sequential



datadir = "./hymenoptera_data/data/sketches_png/"

files = os.listdir(datadir)
categories=[]

for i in files:
    categories.append(i)

print(categories)
num_classes=len(categories)

image_w = 299
image_h = 299

X=[]
Y=[]

for idex, categorie in enumerate(categories):
    label = [0 for i in range(num_classes)]
    label[idex] = 1
    image_dir = datadir + categorie + '/'

    for top, dir, f in os.walk(image_dir):
        for filename in f:
            print(image_dir + filename)
            img = cv2.imread(image_dir + filename, 0)
            #CCL
            img = cv2.threshold(img, 210, 255, cv2.THRESH_BINARY)[1]  # binary image
            num_labels, labels_im = cv2.connectedComponents(img)

            label_hue = np.uint8(100 * labels_im / np.max(labels_im))
            blank_ch = 255 * np.ones_like(label_hue)
            labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

            # cvt to BGR for display
            labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

            # set bg label to black
            labeled_img[label_hue == 0] = 0
            # cv2.imshow('test', labeled_img)

            for h in range(labeled_img.shape[0]):
                for w in range(labeled_img.shape[1]):
                    if ((labeled_img[h][w][0] == 0) & (labeled_img[h][w][1] == 0) & (labeled_img[h][w][1] == 0)):
                        labeled_img[h][w][0] = 255
                        labeled_img[h][w][1] = 255
                        labeled_img[h][w][2] = 255

            img=labeled_img
            img = cv2.resize(img, None, fx=image_w / img.shape[1], fy=image_h / img.shape[0])
            X.append(img / 256)
            Y.append(label)

X = np.array(X)
Y = np.array(Y)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y)
xy = (X_train, X_test, Y_train, Y_test)

model = Sequential()
model.add(InceptionV3(include_top=False, weights='imagenet'))
model.add(layers.GlobalAveragePooling2D())

extractor_output_shape = model.get_output_shape_at(0)[1:]

model.add(layers.InputLayer(input_shape=extractor_output_shape))
model.add(layers.Dense(2048, activation='sigmoid'))

model.summary()

model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
model.fit(X_train, Y_train, batch_size=64, epochs = 10)

model.save('sketch_inceptionv3_ccl.h5')



