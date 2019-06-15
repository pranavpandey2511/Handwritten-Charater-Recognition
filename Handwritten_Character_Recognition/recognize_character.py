import cv2
import numpy as np
from keras.models import model_from_json

def load_model():
    # load json and create model
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("model.h5")
    print("Loaded model from disk")
    return loaded_model

model = load_model()
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])


# creating a 600 x 600 pixels canvas for mouse drawing
canvas = np.ones((600,600), dtype="uint8") * 255
# designating a 400 x 400 pixels point of interest on which digits will be drawn
canvas[100:500,100:500] = 0

start_point = None
end_point = None
is_drawing = False

def draw_line(img,start_at,end_at):
    cv2.line(img,start_at,end_at,255,15)

def on_mouse_events(event,x,y,flags,params):
    global start_point
    global end_point
    global canvas
    global is_drawing
    if event == cv2.EVENT_LBUTTONDOWN:
        is_drawing = True
        ix,iy = x,y

    elif event == cv2.EVENT_MOUSEMOVE:
        if is_drawing == True:
            cv2.circle(canvas,(x,y),10,(0,0,0),-1)

    elif event == cv2.EVENT_LBUTTONUP:
        is_drawing = False
        cv2.circle(canvas,(x,y),10,(0,0,0),-1) 


cv2.namedWindow("Test Canvas")
cv2.setMouseCallback("Test Canvas", on_mouse_events)


while(True):
    cv2.imshow("Test Canvas", canvas)
    key = cv2.waitKey(1) & 0xFF 
    if key == ord('q'):
        break
    elif key == ord('s'):
        is_drawing = True
    elif key == ord('c'):
        canvas[100:500,100:500] = 0
    elif key == ord('p'):
        image = canvas[100:500,100:500]
        result = model.predict(np.array(cv2.resize(image, (28 , 28)).reshape((28 , 28,1)).astype('float32') / 255))
        print("PREDICTION : ",result)

cv2.destroyAllWindows()


