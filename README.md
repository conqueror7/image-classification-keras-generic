# image-classification-keras-generic
Image classification generic code in Keras to support all state of the art classification algorithms.

**Code Explaination**
* `main.py` - Objective of main.py is to start the training process, main.py acts like an abstraction and you can set the parameters for different training configurations.
* `train.py` - Objective of train.py is to set the training configurations and start the training by using any pretrained model as an backend. The models that you can use currently are - 'xception','vgg16','vgg19','resnet50','inceptionv3','inceptionresnetv2','nasnet','densenet','mobilenet'.
* `models.py` - Objective of models.py is to provide an Image Classification architecture for training the model. Please note that currently training is done on the whole network -convolution and fully connected layers. If you want to train only the fully connected (last layer), for example in case of samples similar to imagenet datsets. Then, you can set layers.trainable= False in line of code.
* `test.py` - Objective of test.py to run inference on the model that was created.

**Instructions to run the code**
You can follow the below steps to run the code,
* The `main.py` file has arguments defined which you can use to start the training via command-line or you can edit the default parameters that are set inside the `main.py` file.
* Once that is done, you will have your model saved at every epoch in your models folder.
* You can use the latest weight for testing in the `test.py` file, parameters are defined in the `test.py` file like test_data_path, best_model and validation_data_path to getting the names of the classes.
* A `results.csv` will be generated for all the data points in the test_folder with file_name and class of that image.

**Configuration for training**
* Optimizer - SGD with momentum 
* Loss function - Categorical Cross Entropy
* Dropout - 0.5 at the final Fully Connected layer to avoid overfitting
