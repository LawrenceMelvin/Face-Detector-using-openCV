# FACE DETECTION USING WEBCAM
'''
Before importing openCV, you have install openCV
Go to the command promt
cmd to install openCV: pip install opencv-python
'''
# importing opencv files
import cv2
# importing the trained face data
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# opening the webcam
webcam = cv2.VideoCapture(0)
#Run the webcam frame by frame
while True:
    # read the current frame
    successful_frame_read, frame = webcam.read()
    # convert the colored frame to gray color frame
    grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # detect the coordinate of the face
    face_coordinate = trained_face_data.detectMultiScale(grayscaled_img)
    # face_coordinate return the (x,y) points and width and heigth
    for (x, y, w, h) in face_coordinate:
        # Draw a rectangle using the face_coordinate
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    # Diaplaying the frame
    cv2.imshow('Face-detector', frame)
    # Changes the frame every 1millisecond
    key = cv2.waitKey(1)
    # Q key to quit
    if key == 81 or key == 113:
        break
#Close the video stream
webcam.release()
print("code completed")