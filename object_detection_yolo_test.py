import cv2
import numpy as np

#this implementation did not work very well
#thoughts on training yolo to detect the stop box

def test():
    #yolo = cv2.dnn.readNet('yolov3-tiny.weights','darknet-master/cfg/yolov3-tiny.cfg')

    classes = []

    #r mean read operation
   # with open('darknet-master/data/coco.names', 'r') as f:
        #classes = f.read().splitlines()

    img = cv2.imread('frame_left_undist.png')
    #1/255 ensures values are ranging from 0 to 1
    #resize image on both x and y axis
    blob = cv2.dnn.blobFromImage(img, 1/255, (320,320), (0,0,0), swapRB=True, crop=False)

    yolo.setInput(blob)
    output_layer_name = yolo.getUnconnectedOutLayersNames()
    layer_output = yolo.forward(output_layer_name)

    boxes = []
    confidences = []
    class_ids = []


    height, width, _ = img.shape

    for output in layer_output:
        for detection in output:
            #score is an array of position of boxes and a confidence property of the class being in that box
            score = detection[5:]
            class_id = np.argmax(score)
            confidence = score[class_id]
            #making sure there are no overlapping bounding boxes
            if confidence > 0.7:
                center_x = int(detection[0]*width)
                center_y = int(detection[0] * height)
                w = int(detection[0] * width)
                h = int(detection[0] * height)

                #using x and y values to find the corners
                x = int(center_x - w/2)
                y = int(center_y - h/2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    font = cv2.FONT_HERSHEY_PLAIN
    colors = np.random.uniform(0, 255, size = (len(boxes), 3))

    for i in indexes.flatten():
        x,y,w,h = boxes[i]
        label = str(classes[class_ids[i]])
        confi = str(round(confidences[i]))
        color = colors[i]

        cv2.rectangle(img, (x,y), (x+w, y+h), color, 1)
        cv2.putText(img, label + " " + confi, (x,y+20), font, 2, (255, 255, 255), 2)
        cv2.imshow("test window", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()