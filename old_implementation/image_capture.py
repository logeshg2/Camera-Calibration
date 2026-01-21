# This script is used to capture individual frames for calibration
import cv2

vid = cv2.VideoCapture(0) # give the camera id (eg: 0,1,2,...) 
i = 1
while True:
    check,frame = vid.read()
    if not check:
        print("no camera frames")
        break
    cv2.imshow("out",frame) # frame shape is (480,640,3) by default
    # print(frame.shape)
    key = cv2.waitKey(1)
    if key == ord('s'):
        cv2.imwrite(f".//image_data//img{i}.png",cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)) # save as rgb
        print(f"Saved img{i}.png")
        i+=1
    if key == ord("q"):
        break
vid.release()
cv2.destroyAllWindows()