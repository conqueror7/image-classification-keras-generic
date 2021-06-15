from time import time
from tensorflow.keras import applications
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from tensorflow.keras import backend as k
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
import os

from models import Models
import math


def train_model(data_dir_train,data_dir_valid,batch_size,epochs,model_name,save_loc,weights=None):

    DATA_TRAIN = data_dir_train
    DATA_VALID = data_dir_valid
    BATCH_SIZE = batch_size
    EPOCH = epochs
    num_classes = len(os.listdir(DATA_TRAIN))

    nb_train_samples = sum([len(files) for r, d, files in os.walk(DATA_TRAIN)])
    nb_validation_samples = sum([len(files) for r, d, files in os.walk(DATA_VALID)])

    print("Training Samples :",nb_train_samples)
    print("Validation Samples :",nb_validation_samples)
    print("Number of Classes :",num_classes)


    #we can use any model from the below list of image classification model to train the network
    keras_models= ['xception','vgg16','vgg19','resnet50','inceptionv3','inceptionresnetv2','nasnet','densenet121','densenet169','densenet201','mobilenet','custom']

    modelsBuilder = Models(model_name,num_classes)

    if model_name in keras_models:
        model_final,img_width,img_height = modelsBuilder.createModelBase(weights)


    # Compile the model
    model_final.compile(loss = "categorical_crossentropy", optimizer = optimizers.SGD(lr=0.0001, momentum=0.9), metrics=["accuracy"])
    
    #summary of the model built
    model_final.summary()
    
    # Initiate the train and test generators with data Augumentation
    train_datagen = ImageDataGenerator(
    	rescale = 1./255)

    test_datagen = ImageDataGenerator(
    	rescale = 1./255)

    train_generator = train_datagen.flow_from_directory(
    	DATA_TRAIN,
    	target_size = (img_height, img_width),
    	batch_size = BATCH_SIZE,
    	class_mode = "categorical")

    validation_generator = test_datagen.flow_from_directory(
    	DATA_VALID,
        batch_size = BATCH_SIZE,
    	target_size = (img_height, img_width),
    	class_mode = "categorical")

    # Save the model according to the conditions
    model_save = save_loc + model_name +" weights-{epoch:02d}.hdf5"
    checkpoint = ModelCheckpoint(model_save, monitor='val_accuracy', verbose=1, save_best_only=False, save_weights_only=False, save_freq='epoch')
    early_stopping = EarlyStopping(monitor='val_accuracy', min_delta=0, patience=3, verbose=1, mode='auto', restore_best_weights= True)

    # learning rate schedule
    def step_decay(EPOCH):
    	initial_lrate = 0.0001
    	drop = 0.1
    	epochs_drop = 20.0
    	lrate = initial_lrate * math.pow(drop, math.floor((1+EPOCH)/epochs_drop))
    	print("Epoch: {0:} and Learning Rate: {1:} ".format(EPOCH, lrate))
    	return lrate

    change_lr = LearningRateScheduler(step_decay)

    # Train the model
    model_final.fit_generator(
    	train_generator,
    	steps_per_epoch = nb_train_samples//BATCH_SIZE,
    	epochs = EPOCH,
    	validation_data = validation_generator,
    	validation_steps = nb_validation_samples//BATCH_SIZE,
    	callbacks = [checkpoint, early_stopping, change_lr])


    # Save the final model on the disk
    model_final_name = save_loc + model_name +" weights-final.hdf5"

    if not os.path.exists(os.path.dirname(model_final_name)):
	    try:
	        os.makedirs(os.path.dirname(model_final_name))
	    except OSError as exc: # Guard against race condition
	        if exc.errno != errno.EEXIST:
	            raise

    model_final.save(model_final_name)

    return_string = "Model saved at : "+ model_final_name

    return return_string
