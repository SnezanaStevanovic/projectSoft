import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import dilation, erosion, square, disk, diamond
from skimage.measure import label,regionprops
from vector import distance, pnt2line
from sklearn.datasets import fetch_mldata
from keras.models import Sequential
from keras.layers.core import Activation, Dense
from keras.optimizers import SGD
from keras.models import model_from_json



# mnist = fetch_mldata('MNIST original')

# data   = mnist.data / 255.0
# labels = mnist.target.astype('int')
# train_rank = 10000
# test_rank = 100
# #------- MNIST subset --------------------------
# train_subset = np.random.choice(data.shape[0], train_rank)
# test_subset = np.random.choice(data.shape[0], test_rank)

# # train dataset
# train_data = data[train_subset]
# train_labels = labels[train_subset]

# # test dataset
# test_data = data[test_subset]
# test_labels = labels[test_subset]
# def to_categorical(labels, n):
#     retVal = np.zeros((len(labels), n), dtype='int')
#     ll = np.array(list(enumerate(labels)))
#     retVal[ll[:,0],ll[:,1]] = 1
#     return retVal

# train_out = to_categorical(train_labels, 10)
# test_out = to_categorical(test_labels, 10)

# # prepare model
# model = Sequential()
# model.add(Dense(70, input_dim=784))
# model.add(Activation('relu'))
# model.add(Dense(50))
# model.add(Activation('tanh'))
# model.add(Dense(10))
# model.add(Activation('relu'))

# sgd = SGD(lr=0.1, decay=0.001, momentum=0.7)
# model.compile(loss='mean_squared_error', optimizer=sgd)
# training = model.fit(train_data, train_out, nb_epoch=500, batch_size=400, verbose=0)


# model_json = model.to_json()
# with open("model.json", "w") as json_file:
#     json_file.write(model_json)
# # serialize weights to HDF5
# model.save_weights("model.h5")
# print("Saved model to disk")

json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights("model.h5")
print("Loaded model from disk")
id = -1 
def my_rgb2gray(img_rgb):
    img_gray = np.ndarray((img_rgb.shape[0], img_rgb.shape[1]))
    for i in np.arange(0, img_rgb.shape[0]):
        for j in np.arange(0, img_rgb.shape[1]):
            if img_rgb[i, j, 0] > 240 and img_rgb[i, j, 1] < 140 and img_rgb[i, j, 2] < 140:
                img_gray[i, j] = 255
            else:
                img_gray[i, j] = 0
    img_gray = img_gray.astype('uint8')
    return img_gray

def inRange(r, item, items):
    retVal = []
    for obj in items:
        mdist = distance(item['center'], obj['center'])
        if(mdist<r):
            retVal.append(obj)
    return retVal

def nextId():
    global id
    id += 1
    return id

cap = cv2.VideoCapture("Video/video-0.avi")
frameNumber = 0
begin = []
end=[]
suma=0
kernel = np.ones((2,2),np.uint8)
numbers=[]

while(cap.isOpened()):
    ret, frame = cap.read()
    if(frameNumber == 0):
        img_red = my_rgb2gray(frame)
        img_red_th = img_red > 150
        img_tr_dl = dilation(img_red_th, selem=diamond(3))
        labeled_img = label(img_tr_dl)
        regions = regionprops(labeled_img)
        end = [regions[0].bbox[3],regions[0].bbox[0]]
        begin = [regions[0].bbox[1],regions[0].bbox[2]]
    lower = np.array([230, 230, 230],dtype = "uint8")
    upper = np.array([255 , 255 , 255],dtype = "uint8")
    mask = cv2.inRange(frame, lower, upper)   
    img0 = 1.0*mask
    img0 = cv2.dilate(img0,kernel)
    img0=cv2.erode(img0,kernel)
    img0 = cv2.dilate(img0,kernel)   
    gray_labeled = label(img0)
    regions = regionprops(gray_labeled)
    for region in regions:
        number = {'center' : region.centroid,  'frame' : frameNumber}
        result = inRange(20,number,numbers)
        if len(result) == 0:
            number['id'] = nextId()
            number['pass'] = False
            numbers.append(number)
        elif len(result) == 1:
            result[0]['center'] = number['center']
            result[0]['frame'] = frameNumber
    for el in numbers:
        t = frameNumber - el['frame'] 
        if(t<3):
            dist, pnt, r = pnt2line(el['center'], begin, end)
            if(r>0):
                if(dist<6):
                    if el['pass'] == False:
                        el['pass'] = True
                        blok_size = (28,28)
                        blok_center = el['center']
                        blok_loc = (blok_center[0]-blok_size[0]/2, blok_center[1]-blok_size[1]/2)
                        imgB = img0[blok_loc[0]:blok_loc[0]+blok_size[0],blok_loc[1]:blok_loc[1]+blok_size[1]]
                        imgB_test = imgB.reshape(784)
                        imgB_test = imgB_test/255.
                        tt = model.predict(np.array([imgB_test]), verbose=1)
                        rez_t = tt.argmax(axis=1)
                        suma+= rez_t[0]
                        print suma
    frameNumber+=1
         
           
cap.release()
cv2.destroyAllWindows()
