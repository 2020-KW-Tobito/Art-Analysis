import os
import cv2
import numpy as np

# datadir = "./Sketch_Dataset/"
#
# files = os.listdir(datadir)
# categories=[]
#
# for i in files:
#     categories.append(i)
#
# print(categories)
# num_classes=len(categories)
#
# directory = "Sketch_CCL_fill/"
#
# for c in range(240):
#     file = categories[c]
#     png_file = datadir + file
#
#     img = cv2.imread(png_file, 0)
#     img = cv2.threshold(img, 210, 255, cv2.THRESH_BINARY)[1]  # binary image
#     num_labels, labels_im = cv2.connectedComponents(img)
#
#
#     def imshow_components(labels):
#         # Map component labels to hue val
#         label_hue = np.uint8(100 * labels / np.max(labels))
#         blank_ch = 255 * np.ones_like(label_hue)
#         labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])
#
#         # cvt to BGR for display
#         labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)
#
#         # set bg label to black
#         labeled_img[label_hue == 0] = 0
#         #cv2.imshow('test', labeled_img)
#
#         for h in range(labeled_img.shape[0]):
#             for w in range(labeled_img.shape[1]):
#                 if ((labeled_img[h][w][0] == 0) & (labeled_img[h][w][1] == 0) & (labeled_img[h][w][1] == 0)):
#                     labeled_img[h][w][0] = 255
#                     labeled_img[h][w][1] = 255
#                     labeled_img[h][w][2] = 255
#         # labeled_img = cv2.threshold(labeled_img, , 255, cv2.THRESH_BINARY)[1]
#         #cv2.imshow('anchor png', img)
#         #cv2.imshow('labeled.png', labeled_img)
#         save_file = directory + file
#         cv2.imwrite(save_file, labeled_img)
#         #cv2.waitKey()
#
#
#     imshow_components(labels_im)
#     print(png_file)
img = cv2.imread('test2/house9.png', 0)
img = cv2.threshold(img, 210, 255, cv2.THRESH_BINARY)[1]  # binary image
num_labels, labels_im = cv2.connectedComponents(img)


def imshow_components(labels):
    # Map component labels to hue val
    label_hue = np.uint8(100 * labels / np.max(labels))
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
    # labeled_img = cv2.threshold(labeled_img, , 255, cv2.THRESH_BINARY)[1]
    # cv2.imshow('anchor png', img)
    # cv2.imshow('labeled.png', labeled_img)
    save_file = 'test2/house9_fill.png'
    cv2.imwrite(save_file, labeled_img)
    # cv2.waitKey()


imshow_components(labels_im)