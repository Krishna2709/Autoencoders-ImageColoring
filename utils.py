import numpy as np
import matplotlib.pyplot as plt

from skimage.color import rgb2lab, lab2rgb
from skimage.transform import resize
from skimage.io import imsave

from tensorflow.keras.preprocessing.image import img_to_array, load_img


# Data Preprocessing
def preprocess_data(dataset):

    ''' Convert from RGB to Lab '''
    X =[]
    Y =[]
    for img in dataset[0]:
        try:
            lab = rgb2lab(img)
            X.append(lab[:,:,0]) 
            Y.append(lab[:,:,1:] / 128) # range  [-127, 128] 
        except:
            print(f'Error: Cannot convert from RGB to LAB')
    X = np.array(X)
    Y = np.array(Y)
    X = X.reshape(X.shape+(1,)) #dimensions to be the same for X and Y

    return X, Y


# Preprocess and Coloring the image
def color_image(model, image):

    img1_color=[]

    img1=img_to_array(load_img(image))
    img1 = resize(img1 ,(256,256))
    img1_color.append(img1)
    img1_color = np.array(img1_color, dtype=float)

    img1_color = rgb2lab(1.0/255*img1_color)[:,:,:,0]
    img1_color = img1_color.reshape(img1_color.shape+(1,))

    output1 = model.predict(img1_color)
    output1 = output1*128

    result = np.zeros((256, 256, 3))
    result[:,:,0] = img1_color[0][:,:,0]
    result[:,:,1:] = output1[0]

    imsave("./results/result.png", lab2rgb(result))
