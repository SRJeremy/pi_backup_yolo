
import cv2
import numpy as np
from picamera2 import Picamera2

from ultralytics import YOLO

import copy # for the deep copys

image_w = 240
image_h = 320


# using a model from yolo
#model = YOLO('yolov8n.yaml') #blank yolo from scratch...
#results = model.train(data="coco8.yaml", epochs=100, imgsz=640)
#results = model.train(data="/home/pi/Jeremy/yolo/Rescue_Line_test.v2i.yolov8/data.yaml", epochs=5, imgsz=640)
#model.save("first_model.pt")

#model = YOLO("/home/pi/Jeremy/yolo/model100_gpu_test.pt")
model = YOLO("/home/pi/Jeremy/yolo/model100.pt")
#model = YOLO("/home/pi/Jeremy/yolo/model100.onnx")

# using single camera in python
picam2 = Picamera2()
config = picam2.create_preview_configuration(lores={"size": (640, 480)})
#config = picam2.create_preview_configuration(lores={"size": (320, 240)})
picam2.configure(config)
picam2.start()

while True:
    yuv420 = picam2.capture_array("lores")
    rgb = cv2.cvtColor(yuv420, cv2.COLOR_YUV420p2RGB)
    
    rgb = cv2.resize(rgb, (image_h, image_w))
    
    rgb = cv2.flip(rgb, -1)
    
    rgb = cv2.blur(rgb, (3,3))
    
    cv2.imshow("Camera", rgb)
    
    #display_version = copy.deepcopy(rgb)
    model_version = copy.deepcopy(rgb)
    
    # pass the image into the trained model
    #results = model(rgb) # returns a list
    results = model(model_version)
    #print("results...")
    #print(type(results))
    #print(len(results))
    #print(results)
    
    
    #image = copy.deepcopy(rgb)
    
    for results in results:
        boxes = results.boxes.cpu().numpy() # type 'ultralytics.engine.results.Boxes'
        #print("boxes results: ")
        #print(type(boxes))
        #print(boxes.shape)
        #print("Items found: ")
        #print(boxes.shape[0])
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = box.conf[0]
            class_id = int(box.cls[0])
            
            label = f'{model.names[class_id]} {confidence:.2f}'
            
            cv2.rectangle(rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(rgb, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            #cv2.rectangle(display_version, (x1, y1), (x2, y2), (0, 255, 0), 2)
            #cv2.putText(display_version, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            
    #rgb = cv2.resize(rgb, (640, 480))
    cv2.imshow("classification", rgb)
    cv2.resizeWindow("classification", 640, 480)
    #cv2.imshow("classification", display_version)
    #cv2.imshow("classification", image)
        
    
    
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break
    #print ("test")

cv2.destroyAllWindows()
picam2.stop()

