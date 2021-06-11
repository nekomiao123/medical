import json
import cv2


def test():
    print("I am in utils")


def read_points_from_json_file(path_to_json_file):
    """
    Read only the points array from a json file
    all other attributes are neglected

    :param path_to_json_file: The json file should be in the working directory or a full path of the json file should be given.

    :return: A dictionary array of all points stored in the file. 
            Each point is represented with a dictionary containing a x (width) and a y (height) coordinate based on the pixel position in the image.
    """
    with open(path_to_json_file, 'rb') as file:
        data = json.load(file)
        return data['points']


def read_image(path_to_image_file):
    """
    Read an color image from a file path

    :param path_to_image_file: The image should be in the working directory or a full path to the image should be given.

    :return: The cv2 image object
    """
    return cv2.imread(path_to_image_file)


def visualise_points(image, points_list, color=(255, 255, 0)):
    """
    Draws a set of points on a given image and returns the drawn image

    :param image: Image on which the points are to be drawn
    :param points_list: A list of tuples (x,y)
    :param color: Color of the points to be drawn

    :return: Drawn image
    """
    new_image = image.copy()
    for point in points_list:
        new_image = cv2.drawMarker(
            new_image, (point['x'], point['y']), color, markerType=cv2.MARKER_CROSS, thickness=1)
    return new_image


def interpolate_point(height_original, width_original,
                      height_resized, width_resized, x, y):
    """
    Preserves the relative position of a point on an image
    after resizing the image

    :param height_original: the original height of the image
    :param width_original: the original width of the image
    :param height_resized: the resized height of the image
    :param width_resized: the resized width of the image
    :param x: the x (width) coordinate of the point on the original image
    :param y: the y (height) coordinate of the point on the original image

    :return: the x and y coordinates of the point on the resized image stored as properties in a dictionary
    """
    # calculate the resize factors
    factor_w = width_resized / width_original
    factor_h = height_resized / height_original

    # calculate the new point coordinates
    interpolated_x = round(x * factor_w)
    interpolated_y = round(y * factor_h)

    # handle rounding errors
    if interpolated_x > width_resized - 1:
        interpolated_x = width_resized - 1
    if interpolated_y > height_resized - 1:
        interpolated_y = height_resized - 1
    return {'x': interpolated_x, 'y': interpolated_y}


#######################################################
# Usage examples:

# Specify paths to test helper functions for an image (eg. ".../images/sim_000000.png")
# and the corresponding json file (eg. ".../point_labels/sim_000000.json")
image_path = ''
json_path = ''

if(image_path != '' and json_path != ''):

    # read files
    img = read_image(image_path)
    points = read_points_from_json_file(json_path)

    # display the image
    cv2.imshow('Image', img)
    # wait till user closed the image by pressing any key
    cv2.waitKey()

    # visualise the points on the image
    img_with_points = visualise_points(img, points)
    # display the image with points
    cv2.imshow('Image with points', img_with_points)
    # wait till user closed the image by pressing any key
    cv2.waitKey()

    # specify the current and a new image size
    image_height = img.shape[0]
    image_width = img.shape[1]
    new_image_width = 256
    new_image_height = 144

    # resize the image
    new_img = cv2.resize(img, (new_image_width, new_image_height))

    # interpolate the points
    new_points = [interpolate_point(image_height, image_width, new_image_height,
                                    new_image_width, point['x'], point['y']) for point in points]

    # visualise the points on the new image
    new_img_with_points = visualise_points(new_img, new_points)

    # display the new image with points
    cv2.imshow('Resized image with interpolated points', new_img_with_points)

    # wait till user closed the image by pressing any key
    cv2.waitKey()