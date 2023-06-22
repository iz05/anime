import cv2
import numpy as np

# shoelace formula
def area(polygon): # polygon is a list of coordinates
    s = 0
    for i in range(0, len(polygon)):
        s += polygon[i][0] * polygon[(i + 1) % len(polygon)][1]
        s -= polygon[i][1] * polygon[(i + 1) % len(polygon)][0] # shoelace formula
    return 0.5 * s # signed area, could be negative

# tests if a point is within a polygon by triangulation
def point_in_polygon(polygon, point):
    summed_area = 0
    for i in range(0, len(polygon)):
        summed_area += abs(area([point, polygon[i % len(polygon)], polygon[(i + 1) % len(polygon)]]))
    if summed_area == abs(area(polygon)):
        return True
    return False

# computes center of mass of the polygon
# return type: tuple (double, double)
def center_of_mass(polygon):
    return [sum(polygon[i][0] for i in range(0, len(polygon))) / len(polygon), sum(polygon[i][1] for i in range(0, len(polygon))) / len(polygon)]
 
# dilates polygon based on x and y factor
# return type: polygon (list) of (int, int) tuples
def dilate(polygon, x_factor = 1.5, y_factor = 1.5, p = None):
    cx, cy = p if p != None else center_of_mass(polygon)
    new_polygon = []
    for x, y in polygon:
        # round at the end to minimize error
        x_new, y_new = int(x_factor * (x - cx) + cx), int(y_factor * (y - cy) + cy)
        new_polygon.append((x_new, y_new))
    return new_polygon

# translates polygon based on x and y offset
# return type: polygon (list) of (int, int) tuples
def translate(polygon, x_offset, y_offset):
    new_polygon = []
    for x, y in polygon:
        x_new, y_new = int(x + x_offset), int(y + y_offset)
        new_polygon.append((x_new, y_new))
    return new_polygon

def draw_point(point, img):
    x1, y1 = int(point[0]), int(point[1])
    cv2.circle(img, (x1, y1), 5, (255, 255, 255), -1)

def draw_polygon(polygon, img):
    for i in range(0, len(polygon)):
        x1, y1, x2, y2 = int(polygon[i][0]), int(polygon[i][1]), int(polygon[(i + 1) % len(polygon)][0]), int(polygon[(i + 1) % len(polygon)][1])
        draw_point((x1, y1), img)
        cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 2)

def eye_dimensions(eye_polygon):
    x1 = eye_polygon[0][0] # corner - used for width
    x2 = eye_polygon[3][0] # corner - used for width
    y3 = eye_polygon[2][1] # used for height
    y4 = eye_polygon[4][1] # used for height
    width = abs(x1 - x2)
    height = abs(y3 - y4)
    return width, height

def nose_dimensions(nose_polygon):
    width = abs(nose_polygon[8][0] - nose_polygon[4][0])
    height = abs(nose_polygon[6][1] - nose_polygon[0][1])
    return width, height

# calculate the slope of a best-fit line
def slope(l):
    x = np.array([item[0] for item in l])
    y = np.array([item[1] for item in l])
    a, b = np.polyfit(x, y, 1)
    return a

def tester():
    l = [[0, 0], [1, 1], [2, 2], [3, 3]]
    print(slope(l))
