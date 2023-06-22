import cv2
from source.cartoonize import Cartoonizer
import os
import sys
from source.geometry import *
import numpy as np
import csv
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
import math

class Feature():
    def __init__(self, trans_x, trans_y, dilate_x, dilate_y, thresh):
        self.tx = trans_x # delta x / width(polygon), assumes proportionality
        self.ty = trans_y # delta y / height(polygon), assumes proportionality
        self.dx = dilate_x # desired_width / width(polygon) 
        self.dy = dilate_y # desired_height / height(polygon)
        self.th = thresh
    
    def move(self, polygon, width, height):
        return translate(polygon, width * self.tx, height * self.ty)
    
    def wipe(self, res, polygon, kx, ky, p, d = None):
        new_polygon = dilate(polygon, kx, ky, p)
        if d == None:
            d = res[new_polygon[4][0], new_polygon[4][1]]
        # blue, green, red = int(color[0]), int(color[1]), int(color[2])
        blue, green, red = d
        cv2.fillPoly(res, pts = np.int32([generate_contours(new_polygon)]), color = (blue, green, red))
        # mask = np.zeros(res.shape, dtype = np.uint8)
        # cv2.fillPoly(mask, pts = np.int32([generate_contours(new_polygon)]), color = (255, 255, 255))
        # mask = cv2.GaussianBlur(mask, (21, 21), cv2.BORDER_DEFAULT)
        # cv2.imshow("mask", mask)
        # cv2.waitKey(0)
        # # combining two images
        # final = np.zeros(res.shape, np.float32)
        # shape = res.shape
        # # iterate through each pixel
        # for i in range(0, shape[0]):
        #     for j in range(0, shape[1]):
        #         b, g, r = res[i, j, 0], res[i, j, 1], res[i, j, 2]
        #         final[i, j, 0] = int((1 - mask[i, j, 0] / 255) * b + mask[i, j, 0] / 255 * blue)
        #         final[i, j, 1] = int((1 - mask[i, j, 0] / 255) * g + mask[i, j, 0] / 255 * green)
        #         final[i, j, 2] = int((1 - mask[i, j, 0] / 255) * r + mask[i, j, 0] / 255 * red)
        # cv2.imshow("final", final)
        # cv2.waitKey(0)
        return res

    def place(self, res, feature_image, polygon, width, height, p, resize_width = None, resize_height = None, align_x = "center", align_y = "center"):
        if resize_width == None:
            resize_width, resize_height = int(width * self.dx), int(height * self.dy)
        feature_image = cv2.resize(feature_image, (resize_width, resize_height))
        cx, cy = p
        x_offset = self.calc_offset(cx, feature_image.shape[1], align_x)
        y_offset = self.calc_offset(cy, feature_image.shape[0], align_y)
        overlay(res, feature_image, x_offset, y_offset)
        return res
    
    def calc_offset(coord, length, align):
        if align == "center":
            return int(coord - length // 2)
        elif align == "bottom":
            return int(coord - length)
        elif align == "top":
            return int(coord)

def generate_contours(polygon):
    contours = []
    for tup in polygon:
        contours.append(list(tup))
    return np.array(contours)

def wipe(res, eye_right, eye_left):
    # wiping out old features
    new_eye_right = dilate(eye_right, 3, 4.5)
    new_eye_left = dilate(eye_left, 3, 4.5)

    # finding skin color
    color = res[new_eye_right[0][0], new_eye_right[0][1]]
    blue = int(color[0])
    green = int(color[1])
    red = int(color[2])

    # creating a mask
    mask = np.zeros(res.shape, dtype = np.uint8)
    cv2.fillPoly(mask, pts = np.int32([generate_contours(new_eye_right)]), color = (255, 255, 255))
    cv2.fillPoly(mask, pts = np.int32([generate_contours(new_eye_left)]), color = (255, 255, 255))
    mask = cv2.GaussianBlur(mask, (21, 21), cv2.BORDER_DEFAULT)
    cv2.imwrite("mask.png", mask)

    # cv2.fillPoly(res, pts = np.int32([self.generate_contours(new_eye_right)]), color = (blue, green, red))
    # cv2.fillPoly(res, pts = np.int32([self.generate_contours(new_eye_left)]), color = (blue, green, red))

    # combining two images
    final = np.zeros(res.shape, np.float32)
    shape = res.shape
    # iterate through each pixel
    for i in range(0, shape[0]):
        for j in range(0, shape[1]):
            b, g, r = res[i, j, 0], res[i, j, 1], res[i, j, 2]
            final[i, j, 0] = int((1 - mask[i, j, 0] / 255) * b + mask[i, j, 0] / 255 * blue)
            final[i, j, 1] = int((1 - mask[i, j, 0] / 255) * g + mask[i, j, 0] / 255 * green)
            final[i, j, 2] = int((1 - mask[i, j, 0] / 255) * r + mask[i, j, 0] / 255 * red)
    
    return final

def overlay(large_image, small_image, x_offset, y_offset):
    y1, y2 = y_offset, y_offset + small_image.shape[0]
    x1, x2 = x_offset, x_offset + small_image.shape[1]
    alpha_s = small_image[:, :, 3] / 255.0
    alpha_l = 1.0 - alpha_s
    for c in range(0, 3):
        large_image[y1:y2, x1:x2, c] = (alpha_s * small_image[:, :, c] + alpha_l * large_image[y1:y2, x1:x2, c])

def process():

    # ORIGINAL CODE
    # algo = Cartoonizer(dataroot='damo/cv_unet_person-image-cartoon_compound-models')
    # print("loaded")
    # img = cv2.imread('test_images/' + sys.argv[1])[...,::-1]
    # print("read image")
    # result = algo.cartoonize(img)
    # print("finished cartoonizing")
    # cv2.imwrite('res.png', result)
    # print('finished!')
    
    # VIDEO PROCESSING
    video = cv2.VideoCapture('test_videos/test2.MOV')
    if video.isOpened() == False:
        print("Error opening video")
    print("This video has %s frames." % int(video.get(cv2.CAP_PROP_FRAME_COUNT)))

    # cartoonizing each frame
    frames = []
    frame_count = 0
    algo = Cartoonizer(dataroot='damo/cv_unet_person-image-cartoon_compound-models')

    rpolygons = []
    lpolygons = []
    npolygons = []
    rcenters = []
    lcenters = []
    ncenters = []
    rdimensions = []
    ldimensions = []
    ndimensions = []

    while(video.isOpened()):
        print("Cartoonizing frame %s" % frame_count)
        ret, frame = video.read()
        if ret == True:
            # storing polygon data in csv files
            f = open('test_videos/eye_right.csv', 'a', newline = '')
            writer_r = csv.writer(f)
            g = open('test_videos/eye_left.csv', 'a', newline = '')
            writer_l = csv.writer(g)

            # cartoonizing
            cv2.imwrite("test_videos/frames_raw/frame" + str(frame_count) + ".png", frame)
            frame, eye_right, eye_left, nose, r, l, n = algo.cartoonize(frame)

            if frame_count == 0:
                print(eye_right)
                print(eye_left)
                print(r) 
                print(l)
                print(center_of_mass(eye_right))
                print(center_of_mass(eye_left))
                print(eye_dimensions(eye_right))
                print(eye_dimensions(eye_left))
            
            frames.append(frame)
            rpolygons.append(eye_right)
            lpolygons.append(eye_left)
            npolygons.append(nose)
            rcenters.append(center_of_mass(eye_right))
            lcenters.append(center_of_mass(eye_left))
            ncenters.append(nose[6]) # bottom-most point
            rdimensions.append(eye_dimensions(eye_right))
            ldimensions.append(eye_dimensions(eye_left))
            ndimensions.append(nose_dimensions(nose))

            # modifying csv
            writer_r.writerow(eye_right)
            writer_l.writerow(eye_left)
            f.close()
            g.close()

            # saving cartoonized frame
            cv2.imwrite("test_videos/frames_cartoonized/frame" + str(frame_count) + ".png", frame)
        else:
            break
        frame_count += 1

    # READING CSV FILES
    # f = open('test_videos/eye_right.csv', 'r')
    # reader = csv.reader(f)
    # eyes_right = []
    # for row in reader:
    #     new_row = []
    #     for item in row:
    #         x = item[1 : len(item) - 1].split(".")
    #         new_item = [int(x[0]), int(x[1])]
    #         new_row.append(new_item)
    #     eyes_right.append(new_row)
    # pr = eyes_right[0]
    # print(pr)
    # print(eye_dimensions(pr)) # width, height
    # print(center_of_mass(pr))
    # f.close()  

    # f = open('test_videos/eye_left.csv', 'r')
    # reader = csv.reader(f)
    # eyes_left = []
    # for row in reader:
    #     new_row = []
    #     for item in row:
    #         x = item[1 : len(item) - 1].split(".")
    #         new_item = [int(x[0]), int(x[1])]
    #         new_row.append(new_item)
    #     eyes_left.append(new_row)
    # # pl = eyes_left[0]
    # # print(pl)
    # # print(eye_dimensions(pl)) # width, height
    # # print(center_of_mass(pl))
    # f.close() 

    # # smoothing
    rcenters = smooth(rcenters, 1, 10)
    lcenters = smooth(lcenters, 1, 10)
    # rcenters = lin_smooth(rcenters, 11)
    # lcenters = lin_smooth(lcenters, 11)
    # print(rcenters)

    # CALIBRATION (MANUAL)
    # large_image = cv2.imread('test_videos/frames_cartoonized/frame0.png')
    # right = cv2.imread("feature_images/right_eye.png", cv2.IMREAD_UNCHANGED)
    # left = cv2.imread("feature_images/left_eye.png", cv2.IMREAD_UNCHANGED)
    # width = int(right.shape[1] * 16 / 100) # adjust these parameters
    # height = int(right.shape[0] * 16 / 100) # adjust these parameters
    # right = cv2.resize(right, (width, height))
    # left = cv2.resize(left, (width, height))
    # # cv2.imwrite("img.png", right)
    # overlay(large_image, right, 242, 70) # adjust these parameters
    # overlay(large_image, left, 288, 70) # adjust these parameters
    # cv2.imwrite("overlay_test.png", large_image)

    # print("Width: %s, Height: %s" % (width, height))
    # print("Right eye: polygon is %s, center of mass is %s, (width, height) = %s" % (rpolygons[0], center_of_mass(rpolygons[0]), eye_dimensions(rpolygons[0])))
    # print("Left eye: polygon is %s, center of mass is %s, (width, height) = %s" % (lpolygons[0], center_of_mass(lpolygons[0]), eye_dimensions(lpolygons[0])))

    # FEATURE CLASS INSTANCES
    r = Feature(0.269, 0.467, 2.46, 4.8, 0)
    l = Feature(1.08, 0.433, 2.286, 4.8, 0)
    # r = Feature(-0.1618, -0.0833, 1.8824, 4, 0)
    # l = Feature(0.0857, -0.0692, 1.8286, 4, 0)
    n = Feature(0, 0, 0, 0, 0)
    eye_right_image = cv2.imread("feature_images/right_eye.png", cv2.IMREAD_UNCHANGED)
    eye_left_image = cv2.imread("feature_images/left_eye.png", cv2.IMREAD_UNCHANGED)
    nose_right_image = cv2.imread("feature_images/nose_right.png", cv2.IMREAD_UNCHANGED)
    nose_left_image = cv2.imread("feature_images/nose_left.png", cv2.IMREAD_UNCHANGED)
    res = cv2.imread("test_videos/frames_cartoonized/frame0.png")

    # size manipulation and smoothing
    for i in range(0, len(rdimensions)):
        rdimensions[i] = [rdimensions[i][0] * r.dx, rdimensions[i][1] * r.dy]
        ldimensions[i] = [ldimensions[i][0] * l.dx, ldimensions[i][1] * l.dy]
        ndimensions[i] = [ndimensions[i][0] * n.dx, ndimensions[i][1] * n.dy]
    rdimensions = smooth(rdimensions, 10, 10)
    ldimensions = smooth(ldimensions, 10, 10)
    ndimensions = smooth(ndimensions, 20, 20)

    # FRAME CUSTOMIZATION
    frame_count = 0
    video = cv2.VideoCapture('test_videos/test2.MOV')
    out = cv2.VideoWriter('test_videos/output.MOV',cv2.VideoWriter_fourcc('M','J','P','G'), video.get(cv2.CAP_PROP_FPS), (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))))
    default_color = (227, 223, 233)
    k = 3
    while(frame_count < 222):
        print("Processing frame %s" % frame_count)
        # frame processing
        frame = cv2.imread("test_videos/frames_cartoonized/frame" + str(frame_count) + ".png")
        eye_right, eye_left, nose = rpolygons[frame_count], lpolygons[frame_count], npolygons[frame_count]
        pr, pl, pn = rcenters[frame_count], lcenters[frame_count], ncenters[frame_count]

        # right eye
        # cv2.circle(frame, (int(pr[0]), int(pr[1])), 5, (255, 255, 255), -1)
        width_right, height_right = eye_dimensions(eye_right)
        # eye_right = r.move(eye_right, width_right, height_right)
        if width_right / height_right > 1.6:
            frame = r.wipe(frame, eye_right, 1.5, 2.5, pr, d = default_color)
            frame = r.place(frame, eye_right_image, eye_right, width_right, height_right, pr, int(rdimensions[frame_count][0]), int(rdimensions[frame_count][1]))

        # repeat for left eye
        # cv2.circle(frame, (int(pl[0]), int(pl[1])), 5, (255, 255, 255), -1)
        width_left, height_left = eye_dimensions(eye_left)
        # eye_left = l.move(eye_left, width_left, height_left)
        if width_left / height_left > 1.6:
            frame = l.wipe(frame, eye_left, 1.5, 2.5, pl, d = default_color)
            frame = l.place(frame, eye_left_image, eye_left, width_left, height_left, pl, int(ldimensions[frame_count][0]), int(ldimensions[frame_count][1]))

        # nose
        bridge = nose[ : 4]
        nose_image = None
        if slope(bridge) > 0:
            nose_image = nose_left_image
        else:
            nose_image = nose_right_image
        width_nose, height_nose = nose_dimensions(nose) 
        frame = n.wipe(frame, nose, 1, 1, pn, d = default_color)
        frame = n.place(frame, nose_image, nose, width_nose, height_nose, pn, int(ndimensions[frame_count][0]), int(ndimensions[frame_count][1]), align_y = "bottom")

        cv2.imwrite("test_videos/frames_final/frame" + str(frame_count) + ".png", frame)
        for i in range(0, k):
            out.write(frame)
        frame_count += k

    video.release()
    out.release()

def demo():
    print("ready to start!")
    cam = cv2.VideoCapture(0)
    while True:
        ret, frame = cam.read()
        if not ret:
            print("failed to grab frame")
            break
        cv2.imshow("test", frame)

        k = cv2.waitKey(1)
        if k % 256 == 32:
            img_name = "demo/img.png"
            cv2.imwrite(img_name, frame)
            break
    cam.release()

    # ORIGINAL CODE
    algo = Cartoonizer(dataroot='damo/cv_unet_person-image-cartoon_compound-models')
    img = cv2.imread('demo/img.png')[...,::-1]
    result, left, right = algo.cartoonize(img)
    cv2.imwrite('demo/res.png', result)
    print('finished!')
    res = cv2.imread("demo/res.png")
    cv2.imshow("res", res)
    cv2.waitKey(0)

def smooth(array, x, y):
    refX = array[0][0]
    refY = array[0][1]
    for element in array:
        if element[0] < refX:
            element[0] = refX - x * math.floor((refX - element[0]) / x)
        elif element[0] > refX:
            element[0] = refX + x * math.floor((element[0] - refX) / x)
        if element[1] < refY:
            element[1] = refY - y * math.floor((refY - element[1]) / y)
        elif element[1] > refY:
            element[1] = refY + y * math.floor((element[1] - refY) / y)
    return array

def lin_smooth(array, k):
    for i in range(0, len(array) - k, k):
        x1, y1, x2, y2 = array[i][0], array[i][1], array[i + k][0], array[i + k][1]
        for j in range(1, k):
            array[i + j] = [x1 * j / k + x2 * (k - j) / k, y1 * j / k + y2 * (k - j) / k]
    return array

if __name__ == '__main__':
    process()