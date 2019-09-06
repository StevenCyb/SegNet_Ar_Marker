import os
import random
import numpy as np
import cv2
import cv2.aruco as aruco

# Get dictionary by name
def get_dictionary(DICT_4X4_50=False, DICT_4X4_100=False, DICT_4X4_250=False, DICT_4X4_1000=False, DICT_5X5_50=False, DICT_5X5_100=False, DICT_5X5_250=False, DICT_5X5_1000=False, DICT_6X6_50=False, DICT_6X6_100=False, DICT_6X6_250=False, DICT_6X6_1000=False, DICT_7X7_50=False, DICT_7X7_100=False, DICT_7X7_250=False, DICT_7X7_1000=False, DICT_ARUCO_ORIGINAL=False, DICT_APRILTAG_16h5=False, DICT_APRILTAG_25h9=False, DICT_APRILTAG_36h10=False, DICT_APRILTAG_36h11=False):
    dictionary = []
    if DICT_4X4_50:
        dictionary.append([aruco.DICT_4X4_50, 50])
    if DICT_4X4_100:
        dictionary.append([aruco.DICT_4X4_100, 100])
    if DICT_4X4_250:
        dictionary.append([aruco.DICT_4X4_250, 250])
    if DICT_4X4_1000:
        dictionary.append([aruco.DICT_4X4_1000, 1000])
    if DICT_5X5_50:
        dictionary.append([aruco.DICT_5X5_50, 50])
    if DICT_5X5_100:
        dictionary.append([aruco.DICT_5X5_100, 100])
    if DICT_5X5_250:
        dictionary.append([aruco.DICT_5X5_250, 250])
    if DICT_5X5_1000:
        dictionary.append([aruco.DICT_5X5_1000, 1000])
    if DICT_6X6_50:
        dictionary.append([aruco.DICT_6X6_50, 50])
    if DICT_6X6_100:
        dictionary.append([aruco.DICT_6X6_100, 100])
    if DICT_6X6_250:
        dictionary.append([aruco.DICT_6X6_250, 250])
    if DICT_6X6_1000:
        dictionary.append([aruco.DICT_6X6_1000, 1000])
    if DICT_7X7_50:
        dictionary.append([aruco.DICT_7X7_50, 50])
    if DICT_7X7_100:
        dictionary.append([aruco.DICT_7X7_100, 100])
    if DICT_7X7_250:
        dictionary.append([aruco.DICT_7X7_250, 250])
    if DICT_7X7_1000:
        dictionary.append([aruco.DICT_7X7_1000, 1000])
    if DICT_ARUCO_ORIGINAL:
        dictionary.append([aruco.DICT_ARUCO_ORIGINAL, 1024])
    if DICT_APRILTAG_16h5:
        dictionary.append([aruco.DICT_APRILTAG_16h5, 30])
    if DICT_APRILTAG_25h9:
        dictionary.append([aruco.DICT_APRILTAG_25h9, 35])
    if DICT_APRILTAG_36h10:
        dictionary.append([aruco.DICT_APRILTAG_36h10, 2320])
    if DICT_APRILTAG_36h11:
        dictionary.append([aruco.DICT_APRILTAG_36h11, 587])
    return dictionary

# Generate a random marker with random id 
def get_random_marker(shape=(512,512,3), DICT_4X4_50=False, DICT_4X4_100=False, DICT_4X4_250=False, DICT_4X4_1000=False, DICT_5X5_50=False, DICT_5X5_100=False, DICT_5X5_250=False, DICT_5X5_1000=False, DICT_6X6_50=False, DICT_6X6_100=False, DICT_6X6_250=False, DICT_6X6_1000=False, DICT_7X7_50=False, DICT_7X7_100=False, DICT_7X7_250=False, DICT_7X7_1000=False, DICT_ARUCO_ORIGINAL=False, DICT_APRILTAG_16h5=False, DICT_APRILTAG_25h9=False, DICT_APRILTAG_36h10=False, DICT_APRILTAG_36h11=True):
    # Create dictionary list with all enabled families
    dictionary = get_dictionary(DICT_4X4_50, DICT_4X4_100, DICT_4X4_250, DICT_4X4_1000, DICT_5X5_50, DICT_5X5_100, DICT_5X5_250, DICT_5X5_1000, DICT_6X6_50, DICT_6X6_100, DICT_6X6_250, DICT_6X6_1000, DICT_7X7_50, DICT_7X7_100, DICT_7X7_250, DICT_7X7_1000, DICT_ARUCO_ORIGINAL, DICT_APRILTAG_16h5, DICT_APRILTAG_25h9, DICT_APRILTAG_36h10, DICT_APRILTAG_36h11)
    dictionary = random.choice(dictionary)
    # Generate the marker with a white border
    marker = aruco.drawMarker(aruco.getPredefinedDictionary(dictionary[0]), random.randint(0, dictionary[1]-1), int(shape[0] / 2) - 1)
    marker = cv2.copyMakeBorder(marker, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=255)
    # Convert to bgr to have three channels
    marker = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)
    # Change the marker color with random ofset
    marker[np.where((marker==[255,255,255]).all(axis=2))] = random.randint(245, 255)
    marker[np.where((marker==[0,0,0]).all(axis=2))] = random.randint(0, 15)
    return marker

# Put the marker randomly on the background
def generate_sample_with_given_background(background, marker):
    # Resize the marker randomly
    marker = cv2.resize(marker, (int(marker.shape[1] * (random.randint(3, 9)/10)),
                     int(marker.shape[0] * (random.randint(3, 9)/10))), interpolation=cv2.INTER_AREA)
    height, width, _ = marker.shape

    # Make a random perspective transformation
    marker = cv2.warpPerspective(marker,
                   cv2.getPerspectiveTransform(
                    np.float32([[0, 0], [width, 0], [0, height], [width, height]]),
                    np.float32([[(random.randint(0,10)/100)*width, (random.randint(0,10)/100)*height],
                        [width-((random.randint(0,10)/100)*width), (random.randint(0,10)/100)*height],
                        [(random.randint(0,10)/100)*width, height-((random.randint(0,10)/100)*height)],
                        [width-((random.randint(0,10)/100)*width), height-((random.randint(0,10)/100)*height)]])
                   ),
                   (width, height), borderValue=(1, 1, 1))

    # Rotate randomly
    angle_rot = random.randint(0, 360)
    marker = cv2.warpAffine(marker,
                cv2.getRotationMatrix2D((width/2, height/2), np.random.uniform(angle_rot)-angle_rot/2,1),
                (width, height), borderValue=(1, 1, 1))

    # Put the marker on a new image with transparent background equal to one
    marker2 = np.ones(shape=background.shape).astype(np.uint8)
    x = random.randint(0, background.shape[1] - marker.shape[1])
    y = random.randint(0, background.shape[0] - marker.shape[0])
    marker2[y:y+marker.shape[0], x:x+marker.shape[1]] = marker[0:marker.shape[0], 0:marker.shape[1]]

    # Generate the ground through mask
    mask = np.zeros(shape=(background.shape[0], background.shape[1], background.shape[2])).astype(np.uint8)
    mask[np.where((marker2==[1,1,1]).all(axis=2))] = 255
    mask = 255 - mask
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))

    # Put the non transparent background (marker) on the background image
    idx = (mask == 255)
    background[idx]= marker2[idx]
    
    # Create random illuminations
    illuminationMask = np.zeros(shape=background.shape, dtype=np.uint8)
    shape = background.shape
    for _ in range(random.randint(0, 5)):
        randomNo = random.randint(5,80)
        cv2.circle(illuminationMask, (random.randint(1, shape[1]), random.randint(1, shape[0])),
            int((shape[1] + shape[0]) * (random.randint(1, 20)/100)),
            (randomNo,randomNo,randomNo), -1)
    for _ in range(random.randint(0, 9)):
        randomNo = random.randint(5,80)
        cv2.line(illuminationMask, (random.randint(1, shape[1]), random.randint(1, shape[0])),
            (random.randint(1, shape[1]), random.randint(1, shape[0])),
            (randomNo,randomNo,randomNo),
            random.randint(3, int((shape[1] + shape[0]) * (random.randint(1, 10)/100))))
    illuminationMask = cv2.GaussianBlur(illuminationMask, (35,35), 0)
    if random.randint(0,1) == 0:
        background = cv2.subtract(background, illuminationMask)
    else:
        background = cv2.addWeighted(background, 1.0, illuminationMask, 0.5, 0)

    return background, mask

# Generate sample with an random background
def generate_random_full_synthetic_sample(shape=(512,512,3), DICT_4X4_50=False, DICT_4X4_100=False, DICT_4X4_250=False, DICT_4X4_1000=False, DICT_5X5_50=False, DICT_5X5_100=False, DICT_5X5_250=False, DICT_5X5_1000=False, DICT_6X6_50=False, DICT_6X6_100=False, DICT_6X6_250=False, DICT_6X6_1000=False, DICT_7X7_50=False, DICT_7X7_100=False, DICT_7X7_250=False, DICT_7X7_1000=False, DICT_ARUCO_ORIGINAL=False, DICT_APRILTAG_16h5=False, DICT_APRILTAG_25h9=False, DICT_APRILTAG_36h10=False, DICT_APRILTAG_36h11=True):
    # Generate an blank image
    background = np.random.randint(255, size=shape, dtype=np.uint8)
    # Draw one to five random circles 
    for _ in range(random.randint(1, 5)):
        cv2.circle(background, (random.randint(1, shape[1]), random.randint(1, shape[0])),
               int((shape[1] + shape[0]) * (random.randint(1, 20)/100)),
               (random.randint(0,255), random.randint(0,255), random.randint(0,255)), -1)
    # Draw two to five random rectangles
    for _ in range(random.randint(2, 5)):
        cv2.rectangle(background, (random.randint(1, shape[1]), random.randint(1, shape[0])),
              (random.randint(1, shape[1]), random.randint(1, shape[0])),
              (random.randint(0,255), random.randint(0,255), random.randint(0,255)), cv2.FILLED)
    # Draw five to 15 random lines
    for _ in range(random.randint(5, 15)):
        cv2.line(background, (random.randint(1, shape[1]), random.randint(1, shape[0])),
             (random.randint(1, shape[1]), random.randint(1, shape[0])),
             (random.randint(0,255), random.randint(0,255), random.randint(0,255)),
             random.randint(3, int((shape[1] + shape[0]) * (random.randint(1, 10)/100))))
    # Generate a marker and  put them randomly on the background image
    return generate_sample_with_given_background(background, get_random_marker(shape, DICT_4X4_50, DICT_4X4_100, DICT_4X4_250, DICT_4X4_1000, DICT_5X5_50, DICT_5X5_100, DICT_5X5_250, DICT_5X5_1000, DICT_6X6_50, DICT_6X6_100, DICT_6X6_250, DICT_6X6_1000, DICT_7X7_50, DICT_7X7_100, DICT_7X7_250, DICT_7X7_1000, DICT_ARUCO_ORIGINAL, DICT_APRILTAG_16h5, DICT_APRILTAG_25h9, DICT_APRILTAG_36h10, DICT_APRILTAG_36h11))

# Generate sample with an random image background
def generate_random_half_synthetic_sample(shape=(512,512,3), DICT_4X4_50=False, DICT_4X4_100=False, DICT_4X4_250=False, DICT_4X4_1000=False, DICT_5X5_50=False, DICT_5X5_100=False, DICT_5X5_250=False, DICT_5X5_1000=False, DICT_6X6_50=False, DICT_6X6_100=False, DICT_6X6_250=False, DICT_6X6_1000=False, DICT_7X7_50=False, DICT_7X7_100=False, DICT_7X7_250=False, DICT_7X7_1000=False, DICT_ARUCO_ORIGINAL=False, DICT_APRILTAG_16h5=False, DICT_APRILTAG_25h9=False, DICT_APRILTAG_36h10=False, DICT_APRILTAG_36h11=True):
    # Load a random image from bg directory
    background = None
    while background is None:
        background = cv2.imread("./bg/" + os.listdir("./bg/")[(random.randint(0, len(os.listdir("./bg/")) - 1))], 3)
    # Resize image to network size
    background = cv2.resize(background, (shape[0], shape[1]))
    # Generate a marker and  put them randomly on the background image
    return generate_sample_with_given_background(background, get_random_marker(shape, DICT_4X4_50, DICT_4X4_100, DICT_4X4_250, DICT_4X4_1000, DICT_5X5_50, DICT_5X5_100, DICT_5X5_250, DICT_5X5_1000, DICT_6X6_50, DICT_6X6_100, DICT_6X6_250, DICT_6X6_1000, DICT_7X7_50, DICT_7X7_100, DICT_7X7_250, DICT_7X7_1000, DICT_ARUCO_ORIGINAL, DICT_APRILTAG_16h5, DICT_APRILTAG_25h9, DICT_APRILTAG_36h10, DICT_APRILTAG_36h11))