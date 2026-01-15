from ImageUtils import parse_record
from DataReader import load_data, train_vaild_split
from Model import Cifar

import os
import argparse

def configure():
    parser = argparse.ArgumentParser()
    ### YOUR CODE HERE
    parser.add_argument("--resnet_version", type=int, default=1, help="the version of ResNet [1 or 2]")
    parser.add_argument("--resnet_size", type=int, default=3, help='n: the size of ResNet-(6n+2) v1 or ResNet-(9n+2) v2')
    parser.add_argument("--batch_size", type=int, default=128, help='training batch size')
    parser.add_argument("--num_classes", type=int, default=10, help='number of classes')
    parser.add_argument("--save_interval", type=int, default=10, help='save the checkpoint when epoch MOD save_interval == 0')
    parser.add_argument("--first_num_filters", type=int, default=16, help='number of filters for the first conv layer')
    parser.add_argument("--weight_decay", type=float, default=2e-4, help='weight decay rate')
    parser.add_argument("--modeldir", type=str, default='model_v1', help='model directory')
    
    # --- Hyperparameters for Tuning ---
    parser.add_argument("--data_dir", type=str, default='./cifar-10-batches-py', help='directory to the CIFAR-10 dataset')
    parser.add_argument("--learning_rate", type=float, default=0.1, help='initial learning rate')
    parser.add_argument("--momentum", type=float, default=0.9, help='momentum for SGD')
    parser.add_argument("--max_epoch", type=int, default=200, help='max epochs for tuning (Step 1)')
    parser.add_argument("--max_epoch_final", type=int, default=200, help='max epochs for final training (Step 2)')
    
    # Define learning rate decay schedule
    parser.add_argument('--lr_decay_epochs', nargs='+', type=int, default=[100, 150],help='Epochs at which to decay learning rate')
    parser.add_argument('--lr_decay_factor', type=float, default=0.1,help='Factor by which to decay learning rate')
    ### YOUR CODE HERE
    return parser.parse_args()

def main(config):
    print("--- Preparing Data ---")

    ### YOUR CODE HERE
    data_dir = "/content/drive/MyDrive/cifar-10-batches-py"
    ### YOUR CODE HERE

    x_train, y_train, x_test, y_test = load_data(data_dir)
    x_train_new, y_train_new, x_valid, y_valid = train_vaild_split(x_train, y_train)

    model = Cifar(config).cuda()

    ### YOUR CODE HERE
    # First step: use the train_new set and the valid set to choose hyperparameters.
    # model.train(x_train_new, y_train_new, 200)
    # model.test_or_validate(x_valid, y_valid, [160, 170, 180, 190, 200])

    # Second step: with hyperparameters determined in the first run, re-train
    # your model on the original train set.
    # model.train(x_train, y_train, 10)

    # Third step: after re-training, test your model on the test set.
    # Report testing accuracy in your hard-copy report.
    # model.test_or_validate(x_test, y_test, [10])
    ### END CODE HERE

if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    config = configure()
    main(config)