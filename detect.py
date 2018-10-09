import cv2
import argparse
import numpy as np
import glob as g

ap = argparse.ArgumentParser()
ap.add_argument('-c', '--config', required=True,
                help = 'path to yolo config file')
ap.add_argument('-w', '--weights', required=True,
                help = 'path to yolo pre-trained weights')
ap.add_argument('-cl', '--classes', required=True,
                help = 'path to text file containing class names')
args = ap.parse_args()


def get_output_layers(net):
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    return output_layers

def detectCars(img, idx):
    
    print("Detecting image " +  str(idx) + "...")

    #grab dimensions of image
    h, w = img.shape[:2]
    scale = 0.00392

    #add class names to a list
    classes = []
    f = open(args.classes, 'r')
    for line in f:
        classes.append(line.strip())

    #read pre trained weights and load the config file
    net = cv2.dnn.readNet(args.weights, args.config)
    blob = cv2.dnn.blobFromImage(img, scale, (416,416), (0,0,0), True, crop=False)
    net.setInput(blob)

    #obtain outputs
    outputs = net.forward(get_output_layers(net))

    class_ids = []
    confidences = []
    boxes = []
    conf_threshold = 0.5
    nms_threshold = 0.4


    for out in outputs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * w)
                center_y = int(detection[1] * h)
                u = int(detection[2] * w)
                v = int(detection[3] * h)
                x = center_x - u / 2
                y = center_y - v / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, u, v])

    #non max suppression
    ind = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    #draw the bounding boxes
    for i in ind:
        i = i[0]
        box = boxes[i]
        if str(classes[class_ids[i]]) == 'car':
            cv2.rectangle(img, 
                        (round(box[0]), round(box[1])), 
                        (round(box[0]) + round(box[2]), round(box[1]) + round(box[3])), 
                        (255, 255, 255), 2)

    #save image
    save_name = str(idx) + "-detected.jpg"
    cv2.imwrite(save_name, img)

image_list = []
for pic in g.glob('images/*.png'):
    i = cv2.imread(pic)
    image_list.append(i)

idx = 0
for img in image_list:
    detectCars(img, idx)
    idx += 1

print("Done")