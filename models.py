# Import keras Libraries
from tensorflow.keras import applications
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dropout, Flatten, Dense
from tensorflow.keras.models import Model, load_model


class Models():

    def __init__(self,model_name,num_classes):

        self.MODELNAME = model_name
        self.NUMBER_OF_CLASSES = num_classes

    def decideModelName(self):
        model_name_for_call = "invalid"

        if (self.MODELNAME.lower() == 'resnet50'):
            model_name_for_call = 'ResNet50_'
        elif (self.MODELNAME.lower() == 'xception'):
            model_name_for_call = 'Xception_'
        elif (self.MODELNAME.lower() == 'vgg16'):
            model_name_for_call = 'VGG16_'
        elif (self.MODELNAME.lower() == 'vgg19'):
            model_name_for_call = 'VGG19_'
        elif (self.MODELNAME.lower() == 'inceptionv3'):
            model_name_for_call = 'InceptionV3_'
        elif (self.MODELNAME.lower() == 'inceptionresnetv2'):
            model_name_for_call = 'InceptionResNetV2_'
        elif (self.MODELNAME.lower() == 'nasnet'):
            model_name_for_call = 'NASNetLarge_'
        elif (self.MODELNAME.lower() == 'densenet121'):
            model_name_for_call = 'DenseNet121_'
        elif (self.MODELNAME.lower() == 'densenet169'):
            model_name_for_call = 'DenseNet169_'
        elif (self.MODELNAME.lower() == 'densenet201'):
            model_name_for_call = 'DenseNet201_'
        elif (self.MODELNAME.lower() == 'mobilenet'):
            model_name_for_call = 'MobileNet_'
        elif (self.MODELNAME.lower() == 'custom'):
            model_name_for_call = 'CustomNet_'
        return model_name_for_call

    def createModelBase(self, weights='imagenet'):

        model_name_for_call  = self.decideModelName()
        modelCall = getattr(self, model_name_for_call)
        model , img_width, img_height = modelCall(weights)
        
        for layer in model.layers:
            print(layer.name)
            layer.trainable = True
        
        if weights == 'imagenet':
            model.layers.pop()
            pretrained_inputs = model.inputs
            model = Flatten()(model.output)
            model = Dense(512,activation='relu')(model)
            model = Dropout(0.5)(model)
            predictions = Dense(self.NUMBER_OF_CLASSES,activation='softmax')(model)
        else:
            print(model.summary())
            pretrained_inputs = model.input
            predictions = Dense(self.NUMBER_OF_CLASSES, activation='softmax', name='dense_1')(model.layers[-2].output)
            print(model.summary())
        
        model_final = Model(inputs=pretrained_inputs, outputs=predictions)
        return model_final , img_height ,img_width

    
    def CustomNet_(self, weights):
        img_width, img_height = 224, 224
        return load_model(weights),img_width, img_height
        
    def ResNet50_(self, weights = "imagenet"):
        img_width, img_height = 224, 224
        return  applications.resnet50.ResNet50(weights=weights, include_top=False, input_shape=(img_width, img_height, 3)) , img_width, img_height

    def Xception_(self, weights = "imagenet"):
        img_width, img_height = 299, 299
        return applications.xception.Xception(weights=weights, include_top=False, input_shape=(img_width, img_height, 3)) , img_width, img_height

    def VGG16_(self, weights = "imagenet"):
        img_width, img_height = 224, 224
        return applications.vgg16.VGG16(weights=weights, include_top=False, input_shape=(img_width, img_height, 3)) , img_width, img_height

    def VGG19_(self, weights = "imagenet"):
        img_width, img_height = 224, 224
        return applications.vgg19.VGG19(weights=weights, include_top=False, input_shape=(img_width, img_height, 3)) , img_width, img_height

    def InceptionV3_(self, weights = "imagenet"):
        img_width, img_height = 299, 299
        return applications.inception_v3.InceptionV3(weights=weights, include_top=False, input_shape = (img_width, img_height, 3)) , img_width, img_height

    def InceptionResNetV2_(self, weights = "imagenet"):
        img_width, img_height = 299, 299
        return applications.inception_resnet_v2.InceptionResNetV2(weights=weights, include_top=False, input_shape=(img_width, img_height, 3)) , img_width, img_height

    def NASNetLarge_(self, weights = "imagenet"):
        img_width, img_height = 331, 331
        return applications.nasnet.NASNetLarge(weights=weights, include_top=False, input_shape=(img_width, img_height, 3)) , img_width, img_height

    def DenseNet121_(self, weights = "imagenet"):
        img_width, img_height = 224, 224
        return applications.densenet.DenseNet121(weights=weights, include_top=False, input_shape=(img_width, img_height, 3)) , img_width, img_height

    def DenseNet201_(self, weights = "imagenet"):
        img_width, img_height = 224, 224
        return applications.densenet.DenseNet201(weights=weights, include_top=False, input_shape=(img_width, img_height, 3)) , img_width, img_height

    def DenseNet169_(self, weights = "imagenet"):
        img_width, img_height = 224, 224
        return applications.densenet.DenseNet169(weights=weights, include_top=False, input_shape=(img_width, img_height, 3)) , img_width, img_height

    def MobileNet_(self, weights = "imagenet"):
        img_width, img_height = 224, 224
        return applications.mobilenet.MobileNet(weights=weights, include_top=False, input_shape=(img_width, img_height, 3)) , img_width, img_height
