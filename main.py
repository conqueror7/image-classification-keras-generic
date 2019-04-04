import argparse
import train

"""

All the recognition models available in keras are:
keras_models= ['xception','vgg16','vgg19','resnet50','inceptionv3','inceptionresnetv2','nasnet','densenet','mobilenet']

"""
def print_arguments():
    print('data_dir_train : ',args.data_dir_train)
    print('data_dir_valid : ',args.data_dir_valid)
    print('model_name : ',args.model_name)
    print('epochs : ',args.epochs)
    print('batch_size :',args.batch_size)
    print('save_loc : ',args.save_loc)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training a Image Classification model')

    parser.add_argument('-dtrain', '--data_dir_train', default="data/train", help = 'Path to Training Data')
    parser.add_argument('-dvalid', '--data_dir_valid', default="data/validation", help = 'Path to Validation Data')
    parser.add_argument('-m', '--model_name', default="resnet50",help = 'Pretrained model name')
    parser.add_argument('-e', '--epochs', default=10, type=int, help = 'Number of epochs')
    parser.add_argument('-b', '--batch_size', default=16, type=int , help = 'Batch-size')
    parser.add_argument('-s', '--save_loc', default="models/" ,help = 'Save location for the trained models')


    args = parser.parse_args()

    #print the arguments we passed to the model
    print_arguments()

    output_string = train.train_model(args.data_dir_train,args.data_dir_valid,args.batch_size,args.epochs,args.model_name,args.save_loc)

    print(output_string)
