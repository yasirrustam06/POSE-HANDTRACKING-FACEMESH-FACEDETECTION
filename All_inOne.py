import cv2
import cvzone

from cvzone.FaceDetectionModule import FaceDetector
from cvzone.HandTrackingModule import HandDetector
from cvzone.FaceMeshModule import FaceMeshDetector
from cvzone.PoseModule import PoseDetector


pose_detector = PoseDetector(detectionCon=0.5,trackCon=0.5)
Fase_meshDetector = FaceMeshDetector(maxFaces=2,minTrackCon=0.5,minDetectionCon=0.5)
Hand_Detector = HandDetector(maxHands=2,detectionCon=0.5,minTrackCon=0.5)
face_detector = FaceDetector(minDetectionCon=0.5)
cap = cv2.VideoCapture("faces.mp4")


while True:

    ret ,img = cap.read()
    Image,faces = face_detector.findFaces(img)
    lmst,HandImage = Hand_Detector.findHands(Image)
    FaceMesh_Image,KeyPoints = Fase_meshDetector.findFaceMesh(HandImage)
    Final_Image = pose_detector.findPose(FaceMesh_Image)
    Final_Image = cv2.resize(Final_Image,(900,550))



    ImgList = [Final_Image, Final_Image, Final_Image,
               Final_Image, Final_Image, Final_Image,
               Final_Image, Final_Image, Final_Image
       ]




    ImageStack = cvzone.stackImages(ImgList,cols=3 , scale=0.4)
    cv2.imshow("Image",ImageStack)
    if cv2.waitKey(1)  & 0xFF == ord("Q"):
        break

cap.release()
cv2.destroyAllWindows()

