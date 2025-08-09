'''
this program allows the user to press SPACE and take a single image, then draw a bounding rectangle wherever they want
'''

'''
This program uses inrange to detect orange objects (silicon wheel) and draw bounding rect and center
'''

import cv2
import numpy as np
from picamera2 import Picamera2

import copy


image_path = "./capture_images_1/"

#image_w = 480 # 240
#image_h = 640 # 320
image_w = 240
image_h = 320

# if __name__ == "__main__":



# using single camera in python
picam2 = Picamera2()
config = picam2.create_preview_configuration(lores={"size": (640, 480)})
#config = picam2.create_preview_configuration(lores={"size": (320, 240)})
#config = picam2.create_preview_configuration(lores={"size": (image_h, image_w)})
picam2.configure(config)
picam2.start()


image_count = 0

while True:
    yuv420 = picam2.capture_array("lores")
    rgb = cv2.cvtColor(yuv420, cv2.COLOR_YUV420p2RGB)
    
    rgb = cv2.resize(rgb, (image_h, image_w))
    
    rgb = cv2.flip(rgb, -1)
    
    #blur the image slightly
    rgb = cv2.blur(rgb, (3,3))
    #data_image = copy.deepcopy(rgb)
    
    
    # where to do the show image??
    cv2.namedWindow("Camera", cv2.WINDOW_NORMAL)
    #cv2.moveWindow("Camera", 0, 0)
    cv2.imshow("Camera", rgb)
    
    
    user_input = cv2.waitKey(1) & 0xFF

      
    if user_input == ord(' '):
        file_name = image_path + "image" + str(image_count) + ".jpg"
        image_count += 1
        cv2.imwrite(file_name, rgb)
        print("File stored: " + file_name)
        
    elif user_input == ord('q'): 
        break
    
    
    
#end while
# clean-up
cv2.destroyAllWindows()
picam2.stop()
 
