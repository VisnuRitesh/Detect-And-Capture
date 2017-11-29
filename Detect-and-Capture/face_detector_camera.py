import cv2

'''
Face Detector 2: Detects face/faces in frames(webcam)
classifier: haarcascade_frontalface_default.xml from OpenCV library
'''

# load classifier
faceCascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')

video_capture = cv2.VideoCapture(0)

num=1

#Specify the number of images to be captured
while num<300:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(30, 30)
    )

    for (x, y, w, h) in faces:
#        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        # To crop images to your face
        crop=gray[y:y+h,x:x+w]
        cr=cv2.resize(crop,(60,80),interpolation=cv2.INTER_LINEAR)
        cv2.imshow('img',cr)
        cv2.imwrite("/home/visnu/vistrain/negatives/priya"+str(num)+".jpg", cr)
        num+=1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
