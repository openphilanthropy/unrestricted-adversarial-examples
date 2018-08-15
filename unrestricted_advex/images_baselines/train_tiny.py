from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import multi_gpu_model
import keras
import tensorflow as tf

from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input, decode_predictions
from mobilenet import MobileNet
import pickle
import scipy.misc

import numpy as np
import os
from multiprocessing import Pool

smodel = ResNet50(input_shape=(224, 224, 3),
                    weights='imagenet', include_top=True)#'imagenet'

model = keras.models.Sequential([
    keras.layers.Lambda(lambda x: tf.image.resize_images(preprocess_input(x), [224, 224]),
			input_shape=(64,64,3)),
    smodel])

model.summary()

parallel_model = multi_gpu_model(model, gpus=2)
#parallel_model = model

d = "/data/tiny-imagenet-200/train/"
all_files_1 = sorted([os.path.join(root,f) for root,_,fs in os.walk(d) for f in fs if 'JPEG' in f])

ids = set([x.split("/")[-3] for x in all_files_1])

def load(img_path):
    img = scipy.misc.imread(img_path)
    if len(img.shape) != 3:
	img = np.stack([img]*3,axis=2)

    h,w,_ = img.shape
    smaller = min(h,w)
    img = img[(h-smaller)//2:(h-smaller)//2+smaller,
              (w-smaller)//2:(w-smaller)//2+smaller]

    img = scipy.misc.imresize(img, (64, 64), 'cubic')
    return np.array(img,dtype=np.float16)

def create_numpy():
    x = Pool(10).map(load, all_files_1)
    #pickle.dump(x, open("/tmp/tinyimagenet.p","wb"))
    #x = pickle.load(open("/tmp/tinyimagenet.p","rb"))
    x = [z[:,:,:3] for z in x]
    np.save("/tmp/tinyimagenet2.npy", np.array(x))
    print("QQ",np.min(x),np.max(x))
create_numpy()
#exit(0)

train_data = np.load("/tmp/tinyimagenet2.npy")#*255
train_labels = [x.split("/")[4] for x in all_files_1]
to_idx = eval(open("imagenet_class_index.json").read())
labs_to_id = dict((v[0],k) for k,v in to_idx.items())
print(labs_to_id)
train_labels = np.array([labs_to_id[x] for x in train_labels])

print(np.max(train_data),np.min(train_data))

print('train data shape', train_data.shape)
print('train labels shape', train_labels.shape)

datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
)

datagen.fit(train_data)



parallel_model.compile(keras.optimizers.SGD(lr=0.001, momentum=0.9, nesterov=True),
		       loss='sparse_categorical_crossentropy',
		       metrics=['accuracy'])

parallel_model.fit_generator(datagen.flow(train_data, train_labels, batch_size=32, shuffle=True),
                             steps_per_epoch=len(train_data) // 32,
                             epochs=100,
                             callbacks=[
                                        keras.callbacks.ModelCheckpoint("/tmp/imagenet_{epoch}.model", save_weights_only=True),
                             ])

