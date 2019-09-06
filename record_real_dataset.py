import os
import argparse
import numpy as np
import cv2
import cv2.aruco as aruco
from copy import deepcopy
import pyrealsense2 as rs
import utils.sample_generator as sg

# Define arguments with there default values
ap = argparse.ArgumentParser()
ap.add_argument("-dp", "--dataset_path", default='./train_dataset', help="Path to store records (default='./train_dataset').")
ap.add_argument("-rsc", "--rs_calibration", default='./calibration_file.npz', help="Path to calibration file (default='./calibration_file.npz').")
ap.add_argument("-gc", "--gamma_correction", required=False, action="store_true", default=False, help="Create additional gamma corrected records.")
ap.add_argument("-mwb", "--marker_with_border", required=False, type=float, default=1.25, help="Marker-Size with border where 1 is the marker without white border (default=1.25).")
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
ap.add_argument("-bs", "--batch_size", required=False, type=int, default=4, help="The batch size (default=4).")
args = vars(ap.parse_args())

# Count trues in array
def count_trues(arr):
	count = 0
	for b in arr:
		if b:
			count += 1
	return count

# Verify the passed parameters
if not os.path.isdir(os.path.dirname(args["dataset_path"])):
    raise Exception("Path to dataset is invalid.")
if not os.path.isfile(args["rs_calibration"]):
    raise Exception("Path to rs-calibration file is invalid.")
if count_trues([args["DICT_4X4_50"], args["DICT_4X4_100"], args["DICT_4X4_250"], args["DICT_4X4_1000"], args["DICT_5X5_50"], args["DICT_5X5_100"], args["DICT_5X5_250"], args["DICT_5X5_1000"], args["DICT_6X6_50"], args["DICT_6X6_100"], args["DICT_6X6_250"], args["DICT_6X6_1000"], args["DICT_7X7_50"], args["DICT_7X7_100"], args["DICT_7X7_250"], args["DICT_7X7_1000"], args["DICT_ARUCO_ORIGINAL"], args["DICT_APRILTAG_16h5"], args["DICT_APRILTAG_25h9"], args["DICT_APRILTAG_36h10"], args["DICT_APRILTAG_36h11"]]) != 1:
	raise Exception("Exactly one dictionary must be set.")

# Set the parameter
dataset_path = args["dataset_path"] + '/'
gamma_versions = args["gamma_correction"]
dictionary = sg.get_dictionary(DICT_4X4_50=args["DICT_4X4_50"], DICT_4X4_100=args["DICT_4X4_100"], DICT_4X4_250=args["DICT_4X4_250"], DICT_4X4_1000=args["DICT_4X4_1000"], DICT_5X5_50=args["DICT_5X5_50"], DICT_5X5_100=args["DICT_5X5_100"], DICT_5X5_250=args["DICT_5X5_250"], DICT_5X5_1000=args["DICT_5X5_1000"], DICT_6X6_50=args["DICT_6X6_50"], DICT_6X6_100=args["DICT_6X6_100"], DICT_6X6_250=args["DICT_6X6_250"], DICT_6X6_1000=args["DICT_6X6_1000"], DICT_7X7_50=args["DICT_7X7_50"], DICT_7X7_100=args["DICT_7X7_100"], DICT_7X7_250=args["DICT_7X7_250"], DICT_7X7_1000=args["DICT_7X7_1000"], DICT_ARUCO_ORIGINAL=args["DICT_ARUCO_ORIGINAL"], DICT_APRILTAG_16h5=args["DICT_APRILTAG_16h5"], DICT_APRILTAG_25h9=args["DICT_APRILTAG_25h9"], DICT_APRILTAG_36h10=args["DICT_APRILTAG_36h10"], DICT_APRILTAG_36h11=args["DICT_APRILTAG_36h11"])[0]

# Initalize intel realsense
pipeline = rs.pipeline()
realsense_cfg = rs.config()
realsense_cfg.enable_stream(rs.stream.color, 1280, 720, rs.format.rgb8, 6)
pipeline.start(realsense_cfg)

# Gamma correction function. From "Example 1" https://www.programcreek.com/python/example/89460/cv2.LUT
def gammaCorrection(rgb, gammaValue): 
	_rgb = rgb.copy()
	if(gammaValue == 0):
		_rgb
	invGamma = 1.0 / (gammaValue + 1)
	table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
	return cv2.LUT(_rgb, table)

# Load the calibration
mtx = []
dist = []
with np.load(args["rs_calibration"]) as X:
	mtx, dist, _, _ = [X[i] for i in ('mtx', 'dist', 'rvecs', 'tvecs')]

# Define the marker rectangle 
size = args["marker_with_border"] / 2
nsize = -1 * size
axis = np.float32([[nsize, size, 0], [size, size, 0], [size, nsize, 0], [nsize, nsize, 0]])
counter = 0

# Now comes the recording part
print("Press [SPACE] to record a sample and [ESC] to close the application:")
while(True):
	# Get frame from realsense, convert to bgr and create a grayscale image for marker detection
	frames = pipeline.wait_for_frames()
	frame = cv2.cvtColor(np.asanyarray(frames.get_color_frame().get_data()), cv2.COLOR_RGB2BGR)
	display_frame = deepcopy(frame)
	frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
	# Detect markers and draw them
	corners, ids, rejectedImgPoints = aruco.detectMarkers(frame_gray, aruco.getPredefinedDictionary(dictionary))
	if ids is not None and len(ids) > 0:
		display_frame = aruco.drawDetectedMarkers(display_frame, corners, ids)
	
	# Create the gt mask
	mask = np.zeros(shape=(display_frame.shape[0], display_frame.shape[1], display_frame.shape[2])).astype(np.uint8)
	if ids is not None and len(ids) > 0:
		for marker in corners:
			rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(marker, 1, mtx, dist)
			imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, mtx, dist)
			imgpts = np.int32(imgpts).reshape(-1, 2)
			mask = cv2.drawContours(mask, [imgpts[:4]], -1, (255, 255, 255), -1)
	
	# Display the current frame with the corresponding mask 
	cv2.imshow("Record", np.hstack((cv2.resize(display_frame, (0,0), fx=0.5, fy=0.5) , cv2.resize(mask, (0,0), fx=0.5, fy=0.5))))
	key = cv2.waitKey(100)

	# Check if the use have pressen [SPACE] or [ESC]
	if ids is not None and len(ids) > 0:
		if key == 32: # Space
			cv2.imwrite(dataset_path + str(counter) + '.jpg', frame)
			cv2.imwrite(dataset_path + str(counter) + '_mask.jpg', mask)
			# Create gamma corrected records if enabled
			if gamma_versions:
				cv2.imwrite(dataset_path + str(counter + 1) + '.jpg', gammaCorrection(frame, -0.5))
				cv2.imwrite(dataset_path + str(counter + 1) + '_mask.jpg', mask)
				cv2.imwrite(dataset_path + str(counter + 2) + '.jpg', gammaCorrection(frame, -0.25))
				cv2.imwrite(dataset_path + str(counter + 2) + '_mask.jpg', mask)
				cv2.imwrite(dataset_path + str(counter + 3) + '.jpg', gammaCorrection(frame, 0.25))
				cv2.imwrite(dataset_path + str(counter + 3) + '_mask.jpg', mask)
				cv2.imwrite(dataset_path + str(counter + 4) + '.jpg', gammaCorrection(frame, 0.5))
				cv2.imwrite(dataset_path + str(counter + 4) + '_mask.jpg', mask)
				counter += 5
			else:
				counter += 1
			counter += 1
			print('Record: ', counter)
	# Interrupt the application
	if key == 27: 
		break
