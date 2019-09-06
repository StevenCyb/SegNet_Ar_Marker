import os
import argparse
from utils.network import Network

# Define arguments with there default values
ap = argparse.ArgumentParser()
ap.add_argument("-sh", "--shape", required=False, type=tuple, default=(512, 512, 3), help="Define the shape of a tile (default=(256,256,3))).")
ap.add_argument("-lr", "--learning_rate", required=False, type=tuple, default=0.0001, help="Define the learning rate (default=0.0001).")
ap.add_argument("-e", "--iterations", required=False, type=int, default=50000, help="No. of training iterations (default=10000).")
ap.add_argument("-s", "--steps", required=False, type=int, default=1, help="No. of steps per iterations (default=1).")
ap.add_argument("-bs", "--batch_size", required=False, type=int, default=4, help="The batch size (default=4).")
ap.add_argument("-w", "--weights", required=False, default='./weights/weights.ckpt', help="Path where to store the weights (default='./weights/weights.ckpt').")
ap.add_argument("-se", "--saving_iterations", required=False, type=int, default=1000, help="In which steps should the weights be stored (default=100).")
ap.add_argument("-c", "--checkpoint", required=False, default='', help="Continue with checkpoint from [...].")
ap.add_argument("-4X4_50", "--DICT_4X4_50", required=False, action="store_true", default=False, help="Include DICT_4X4_50 into dataset.")
ap.add_argument("-4X4_100", "--DICT_4X4_100", required=False, action="store_true", default=False, help="Include DICT_4X4_100 into dataset.")
ap.add_argument("-4X4_250", "--DICT_4X4_250", required=False, action="store_true", default=False, help="Include DICT_4X4_250 into dataset.")
ap.add_argument("-4X4_1000", "--DICT_4X4_1000", required=False, action="store_true", default=False, help="Include DICT_4X4_1000 into dataset.")
ap.add_argument("-5X5_50", "--DICT_5X5_50", required=False, action="store_true", default=False, help="Include DICT_5X5_50 into dataset.")
ap.add_argument("-5X5_100", "--DICT_5X5_100", required=False, action="store_true", default=False, help="Include DICT_5X5_100 into dataset.")
ap.add_argument("-5X5_250", "--DICT_5X5_250", required=False, action="store_true", default=False, help="Include DICT_5X5_250 into dataset.")
ap.add_argument("-5X5_1000", "--DICT_5X5_1000", required=False, action="store_true", default=False, help="Include DICT_5X5_1000 into dataset.")
ap.add_argument("-6X6_50", "--DICT_6X6_50", required=False, action="store_true", default=False, help="Include DICT_6X6_50 into dataset.")
ap.add_argument("-6X6_100", "--DICT_6X6_100", required=False, action="store_true", default=False, help="Include DICT_6X6_100 into dataset.")
ap.add_argument("-6X6_250", "--DICT_6X6_250", required=False, action="store_true", default=False, help="Include DICT_6X6_250 into dataset.")
ap.add_argument("-6X6_1000", "--DICT_6X6_1000", required=False, action="store_true", default=False, help="Include DICT_6X6_1000 into dataset.")
ap.add_argument("-7X7_50", "--DICT_7X7_50", required=False, action="store_true", default=False, help="Include DICT_7X7_50 into dataset.")
ap.add_argument("-7X7_100", "--DICT_7X7_100", required=False, action="store_true", default=False, help="Include DICT_7X7_100 into dataset.")
ap.add_argument("-7X7_250", "--DICT_7X7_250", required=False, action="store_true", default=False, help="Include DICT_7X7_250 into dataset.")
ap.add_argument("-7X7_1000", "--DICT_7X7_1000", required=False, action="store_true", default=False, help="Include DICT_7X7_1000 into dataset.")
ap.add_argument("-ARUCO_ORIGINAL", "--DICT_ARUCO_ORIGINAL", required=False, action="store_true", default=False, help="Include DICT_ARUCO_ORIGINAL into dataset.")
ap.add_argument("-APRILTAG_16h5", "--DICT_APRILTAG_16h5", required=False, action="store_true", default=False, help="Include DICT_APRILTAG_16h5 into dataset.")
ap.add_argument("-APRILTAG_25h9", "--DICT_APRILTAG_25h9", required=False, action="store_true", default=False, help="Include DICT_APRILTAG_25h9 into dataset.")
ap.add_argument("-APRILTAG_36h10", "--DICT_APRILTAG_36h10", required=False, action="store_true", default=False, help="Include DICT_APRILTAG_36h10 into dataset.")
ap.add_argument("-APRILTAG_36h11", "--DICT_APRILTAG_36h11", required=False, action="store_true", default=False, help="Include DICT_APRILTAG_36h11 into dataset.")
args = vars(ap.parse_args())

# Verify the passed parameters
if not isinstance(args["shape"], tuple) or len(args["shape"]) != 3:
    raise Exception("Shape parameter is invalid. Should be something like '(256,256,3)'.")
if not isinstance(args["learning_rate"], float) or args["learning_rate"] <= 0.0:
    raise Exception("Learning rate parameter is invalid. Should be a float bigger than '0.0'.")
if not isinstance(args["iterations"], int) or args["iterations"] < 1:
    raise Exception("Iterations has an invalid value.")
if not isinstance(args["steps"], int) or args["steps"] < 1:
    raise Exception("Steps argument has an invalid value.")
if not isinstance(args["batch_size"], int) or args["batch_size"] < 1:
    raise Exception("Batch size has an invalid value.")
if not os.path.isdir(os.path.dirname(args["weights"])):
    raise Exception("Path to store weights is invalid.")
if not isinstance(args["saving_iterations"], int) or args["saving_iterations"] < 1:
    raise Exception("Saving iterations has an invalid value.")
if not (args["DICT_4X4_50"] or args["DICT_4X4_100"] or args["DICT_4X4_250"] or args["DICT_4X4_1000"] or args["DICT_5X5_50"] or args["DICT_5X5_100"] or args["DICT_5X5_250"] or args["DICT_5X5_1000"] or args["DICT_6X6_50"] or args["DICT_6X6_100"] or args["DICT_6X6_250"] or args["DICT_6X6_1000"] or args["DICT_7X7_50"] or args["DICT_7X7_100"] or args["DICT_7X7_250"] or args["DICT_7X7_1000"] or args["DICT_ARUCO_ORIGINAL"] or args["DICT_APRILTAG_16h5"] or args["DICT_APRILTAG_25h9"] or args["DICT_APRILTAG_36h10"] or args["DICT_APRILTAG_36h11"]):
    raise Exception("At least one marker family need to be enabled.")

# Initalize the segnet network
network = Network(shape=args["shape"], learning_rate=args["learning_rate"])
# Load checkpoint if is setted
if args["checkpoint"] != '':
	network.load_weights(weights_path=args["weights"])
# Start training
network.train_synthetic_data(iterations=args["iterations"], steps=args["steps"], batch_size=args["batch_size"], weights_path=args["weights"], saving_iterations=args["saving_iterations"], DICT_4X4_50=args["DICT_4X4_50"], DICT_4X4_100=args["DICT_4X4_100"], DICT_4X4_250=args["DICT_4X4_250"], DICT_4X4_1000=args["DICT_4X4_1000"], DICT_5X5_50=args["DICT_5X5_50"], DICT_5X5_100=args["DICT_5X5_100"], DICT_5X5_250=args["DICT_5X5_250"], DICT_5X5_1000=args["DICT_5X5_1000"], DICT_6X6_50=args["DICT_6X6_50"], DICT_6X6_100=args["DICT_6X6_100"], DICT_6X6_250=args["DICT_6X6_250"], DICT_6X6_1000=args["DICT_6X6_1000"], DICT_7X7_50=args["DICT_7X7_50"], DICT_7X7_100=args["DICT_7X7_100"], DICT_7X7_250=args["DICT_7X7_250"], DICT_7X7_1000=args["DICT_7X7_1000"], DICT_ARUCO_ORIGINAL=args["DICT_ARUCO_ORIGINAL"], DICT_APRILTAG_16h5=args["DICT_APRILTAG_16h5"], DICT_APRILTAG_25h9=args["DICT_APRILTAG_25h9"], DICT_APRILTAG_36h10=args["DICT_APRILTAG_36h10"], DICT_APRILTAG_36h11=args["DICT_APRILTAG_36h11"])