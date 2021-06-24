import cv2
import numpy as np
from os import listdir
from os.path import isfile, join

# Get the training data we previously made
data_path = './faces/user/' #our main folder
onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path, f))]

#listdir(datapath) is used to list directories/files in the datapath folder/directory.
#f for f in listdir(datapath) means listing all labels in that directory.
#isfile returns either true or false
#join(datapath,f) joins the file in the datapath.
#overall this line means that onlyfiles will store files that are present in the directory mentioned

# Create arrays for training data and labels
Training_Data, Labels = [], []

# Open training images in our datapath
# Create a numpy array for training data
for i, files in enumerate(onlyfiles):                        #Doing transactions one by one on <onlyfiles>
    image_path = data_path + onlyfiles[i]                    #Storing Data Path combined with the image
    images = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)    #Reading the image at the path
    Training_Data.append(np.asarray(images, dtype=np.uint8)) #Appending the training data and converting input to array(asarray)
    Labels.append(i)                                         #Appending Labels array similarly

# Create a numpy array for both training data and labels
Labels = np.asarray(Labels, dtype=np.int32)                  #Converting labels array into numpy format

# Initialize facial recognizer
# model = cv2.face.createLBPHFaceRecognizer()
# NOTE: For OpenCV 3.0 use cv2.face.createLBPHFaceRecognizer()
# pip install opencv-contrib-python
# model = cv2.createLBPHFaceRecognizer()

facemodel  = cv2.face_LBPHFaceRecognizer.create()           #Creating the model 
facemodel.train(np.asarray(Training_Data), np.asarray(Labels)) #Training the model with the Training Data and corresponding label
print("Model trained sucessefully")
#facemodel.save('face_recognition_model.h5')                   #If you want to save the model in your directory
