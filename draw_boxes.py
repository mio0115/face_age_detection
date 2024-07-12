import cv2


def draw_boxes(image, box_coords):
    for box in box_coords:
        image = cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]))

    return image
