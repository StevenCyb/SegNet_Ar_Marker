import os
import argparse
from utils.network import Network

# Define arguments with there default values
ap = argparse.ArgumentParser()
ap.add_argument("-dp", "--dataset_path", required=False, default='./train_dataset', help="Path to the training dataset (default='./train_dataset').")
ap.add_argument("-sh", "--shape", required=False, type=tuple, default=(512, 512, 3), help="Define the shape of a tile (default=(256,256,3))).")
ap.add_argument("-lr", "--learning_rate", required=False, type=tuple, default=0.0001, help="Define the learning rate (default=0.0001).")
ap.add_argument("-i", "--iterations", required=False, type=int, default=50000, help="No. of training iterations (default=10000).")
ap.add_argument("-s", "--steps", required=False, type=int, default=1, help="No. of steps per iterations (default=1).")
ap.add_argument("-bs", "--batch_size", required=False, type=int, default=4, help="The batch size (default=4).")
ap.add_argument("-w", "--weights", required=False, default='./weights/weights.ckpt', help="Path where to store the weights (default='./weights/weights.ckpt').")
ap.add_argument("-si", "--saving_iterations", required=False, type=int, default=1000, help="In which steps should the weights be stored (default=100).")
ap.add_argument("-c", "--checkpoint", required=False, default='', help="Continue with checkpoint from [...].")
args = vars(ap.parse_args())

# Verify the passed parameters
if not os.path.isdir(args["dataset_path"]):
    raise Exception("Path to training dataset is invalid.")
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

# Load data from path
dataset = []
index_count = 0
current_sample = args["dataset_path"] + "/" + str(index_count) + ".jpg"
current_sample_gt = args["dataset_path"] + "/" + str(index_count) + "_gt.jpg"
while os.path.isfile(current_sample) and os.path.isfile(current_sample_gt):
    dataset.append([current_sample, current_sample_gt])
    index_count += 1
    current_sample = args["dataset_path"] + "/" + str(index_count) + ".jpg"
    current_sample_gt = args["dataset_path"] + "/" + str(index_count) + "_gt.jpg"

if len(dataset) <= 0:
    raise Exception("Cannot finde training samples in the directory " + args["dataset_path"] + ".\r\n To get more information take a look at the original github repo.")
    
# Initalize the segnet network
network = Network(shape=args["shape"], learning_rate=args["learning_rate"])
# Load checkpoint if is setted
if args["checkpoint"] != '':
	network.load_weights(weights_path=args["weights"])
# Start training
network.train_real_data(iterations=args["iterations"], steps=args["steps"], batch_size=args["batch_size"], weights_path=args["weights"], saving_iterations=args["saving_iterations"], dataset=dataset)
