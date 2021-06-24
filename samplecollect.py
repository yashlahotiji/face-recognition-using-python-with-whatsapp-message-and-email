import cv2
import numpy as np

face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') #load haar classifier
# Load functions
def face_extractor(img): 
    # Function detects faces and returns the cropped face
    # If no face detected, it returns the input image
    
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) # We convert our color image to grayscale format for faster calculation
    faces = face_classifier.detectMultiScale(gray) # Detecting face
    
    if faces is (): #Empty round brackets represent empty tuple
       return None  #If there are no faces detected return None i.e. resetting it back to original posititon here
    
    for (x,y,w,h) in faces:                 # Cropping all possible captured faces
        cropped_face = img[ x:x+w , y:y+h]  # We store the complete image inside cropped_face array
        # .detectMultiScale() returns x,y,w,h where (x,y) are top-left edge of a rectangle , w is width , h is height
        # faces[0][0] = x ; faces[0][1] = y ; faces[0][2] = w ; faces[0][3] = h
        # x1 = x ; y1 = y ; x2 = x1 + w (as x1 + width = x2) ; y2 = y1 + h (as y1 + height = y2)
        print(type(cropped_face))

    return cropped_face                     #returning cropped_face to img in face_extractor function

cap = cv2.VideoCapture(0)                   #start webcam
count = 0

# Collect 100 samples of your face from webcam input
while True:
    ret, frame = cap.read()
    if face_extractor(frame) is not None:   #If there is cropped image ready and not empty, then true and get inside loop
        count += 1                          #Count increases by 1 if image captured
        face = cv2.resize(face_extractor(frame), (200, 200))     #Image is resized into size (200,200)
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)            #Converting image into grayscale
        file_name_path = './faces/user/' + str(count) + '.jpg'   #Save file in specified directory with unique name
        cv2.imwrite(file_name_path, face)                        #Image is written on that path with name mentioned
        #Put count on images and display live count
        cv2.putText(face, str(count), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
        cv2.imshow('Face Cropper', face) #Show the window
        
    else:
        print("Face not found")
        pass

    if cv2.waitKey(1) == 13 or count == 100: #13 is the Enter Key
        break
        
cap.release() #Close the camera
cv2.destroyAllWindows()    

print("Collecting Samples Complete")
print(frame.shape)
