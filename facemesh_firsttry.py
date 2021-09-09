from fdlite import FaceDetection, FaceDetectionModel
from fdlite.render import Colors, detections_to_render_data, render_to_image 
from PIL import Image
import cv2
import numpy as np

def cv2_to_pil(img): #Since you want to be able to use Pillow (PIL)
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

def pil_to_cv2(img):
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)



detect_faces = FaceLandmark(model_type=FaceDetectionModel.BACK_CAMERA)    
    
cap = cv2.VideoCapture(0)
while(True):
    ret, img = cap.read()
    PIL_img = cv2_to_pil(img)

    faces = detect_faces(PIL_img)
    render_data = detections_to_render_data(faces, bounds_color=Colors.GREEN)
    print(type(render_to_image(render_data, image)))
    
    #rects = find_faces(img, face_model)
    
    
    #cv2.imshow("image", img)
    #if cv2.waitKey(1) & 0xFF == ord('q'):
    #   break
cap.release()
cv2.destroyAllWindows()
