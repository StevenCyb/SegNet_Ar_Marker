import os
import cv2
import argparse
from utils.network import Network

# Define arguments with there default values
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image.")
ap.add_argument("-sh", "--shape", required=False, type=tuple, default=(512, 512, 3), help="Define the shape of a tile (default=(256,256,3))).")
ap.add_argument("-w", "--weights", required=False, default='./weights/weights.ckpt', help="Path to the weights (default='./weights/weights.ckpt').")
ap.add_argument("-o", "--output", required=True, help="Path to save prediction.")
args = vars(ap.parse_args())

# Verify the passed parameters
if not os.path.isfile(args["image"]):
    raise Exception("Path to image is invalid.")
if not isinstance(args["shape"], tuple) or len(args["shape"]) != 3:
    raise Exception("Shape parameter is invalid. Should be something like '(256,256,3)'.")
if not os.path.isdir(os.path.dirname(args["weights"])):
    raise Exception("Path to weights is invalid.")

# Load the image 
image = cv2.cvtColor(cv2.imread(args["image"], 3), cv2.COLOR_BGR2RGB)

# Initalize the segnet network
network = Network(shape=args["shape"])
# Load the weights
network.load_weights(weights_path=args["weights"])
# Start prediction and save the results
prediction = network.predict(image)
cv2.imwrite(args["output"], cv2.cvtColor(prediction, cv2.COLOR_RGB2BGR))