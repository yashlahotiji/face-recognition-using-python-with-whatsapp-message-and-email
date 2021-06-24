import cv2
import numpy as np
import os
import time
import pywhatkit
import smtplib
import threading

def smes():
    # Sending Whatsapp Message
    pywhatkit.sendwhatmsg_instantly(phone_no="+91 RECEIVER NUMBER", 
                    message="Hi, I am Yash this is whatsapp message sent with Python.")
            
    # Sending email
    print("Whatsapp Message sent Successfully!!")
    pywhatkit.send_mail(email_sender= "SENDER EMAIL",
                    password= "SENDER PASSWORD",
                    subject="Automated E-mail",
                    message="System generated mail using OpenCV face detection.",
                    email_receiver="EMAIL RECEIVED")

def linst():
    # Creating an ec2 instance on aws cloud
    os.system("aws ec2 run-instances  --image-id ami-011c99152163a87ae --instance-type t2.micro  --subnet-id subnet-06f7036d  --count 1 --security-group-ids sg-5771be2b > ec2.txt")
    print("Instance Launched")
            
    # Creating volume of size 5gb
    os.system("aws ec2 create-volume --availability-zone ap-south-1a --size 5 --volume-type gp2 --tag-specification  ResourceType=volume,Tags=[{Key=face,Value=volume}]  > ebs.txt")
    print("Volume Created of size 5 gb")
    print("Initiating in 120 seconds")
    time.sleep(120)
    ec2_id = open("ec2.txt", 'r').read().split(',')[3].split(':')[1].split('"')[1]
    ebs_id = open("ebs.txt", 'r').read().split(',')[6].split(':')[1].split('"')[1]
            
    os.system("aws ec2 attach-volume --instance-id   " + ec2_id +"  --volume-id  " + ebs_id  +"  --device /dev/sdf")
    print("Volume Successfully attached to the instance")
    
send_message_t1 = threading.Thread(target=smes)
launch_instance_t1 = threading.Thread(target=linst)
face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def face_detector(img, size=0.5):
    
    # Convert image to grayscale
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    if faces is ():           #if faces returns no value, return the same image screen available
        return img, []        #return image and empty list since we have no values of x,y,w,h2
    
    #creating a rectangle on the border of the face
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,255),2)
        roi = img[y:y+h, x:x+w]
        roi = cv2.resize(roi, (200, 200))
    return img, roi           #return image and the cropped photograph

# Open Webcam
cap = cv2.VideoCapture(0)

while True:
    launch_instance = False
    send_messages = False
    ret, frame = cap.read()
    
    image, face = face_detector(frame) #function called
    
    try:
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

        # Pass face to prediction model
        # "results" comprises of a tuple containing the label and the confidence value
        results = facemodel.predict(face) #prediction using the cropped image
        print(results)
        # harry_model.predict(face)
        
        #if results[1] < 500:
        confidence = int( 100 * (1 - (results[1])/400) )                #formula to find confidence score
        display_string = str(confidence) + '% Confident it is User'     #variable declared to store confidence percentage
        print(confidence)
        
        cv2.putText(image, display_string, (100, 120), cv2.FONT_HERSHEY_COMPLEX, 1, (255,120,150), 2)
        
        if confidence > 80:
            cv2.putText(image, "Hey Yash", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
            cv2.imshow('Face Recognition', image )
            send_messages = True
            send_message_t1.start()
            from twilio.rest import Client 
 
            account_sid = 'ACCOUNT SID' 
            auth_token = 'ACCOUNT TOKEN' 
            client = Client(account_sid, auth_token) 
 
            message = client.messages.create( 
                              from_='whatsapp:+911234567890',  
                              body='hi',      
                              to='whatsapp:+911234567890'
                                            )
            print(message.sid)
            break
 
            
            # os.system("chrome https://www.google.com/)
            # os.system("wmplayer   c:\.filename.mp3")
            #break
         
        else:
            print("Second face detected - Initiating aws ec2 instance...")
            cv2.destroyAllWindows()
            launch_instance = True
            launch_instance_t1.start()
            image2 = image
            cv2.putText(image2, "Second face detected", (50, 50) , cv2.FONT_HERSHEY_TRIPLEX, 1, (0,255,255), 2)
            cv2.putText(image2, "Initiating aws ec2 instance...", (50, 100), cv2.FONT_HERSHEY_TRIPLEX, 1, (0,255,255), 2)
            cv2.imshow('Second Face Detected', image2 )
            cv2.waitKey(20000)
            cv2.destroyAllWindows()
            break
            
        if launch_instance == True or send_message == True:
            break
            

    except:
        cv2.putText(image, "No Face Found", (220, 120) , cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)
        cv2.putText(image, "looking for face", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)
        cv2.imshow('Face Recognition', image )
        pass
        
        
    if cv2.waitKey(1) == 13: #13 is the Enter Key
        break
        
cap.release()
cv2.destroyAllWindows()
