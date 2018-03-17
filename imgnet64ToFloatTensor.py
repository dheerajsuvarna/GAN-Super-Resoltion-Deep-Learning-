import pickle
import numpy as np
import os
import torch
import torch.utils.data as data


class ImageNet64Data(data.Dataset):

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __getitem__(self, index):
        img = self.X[index]

        img = torch.from_numpy(img)
        return img, None

    def __len__(self):
        return len(self.y)


def load_data_train(root, img_size=64):
    '''
    loads the data from files and returns a torch dataset
    '''
    data_folders = [os.path.join(root, 'Imagenet64_train_part1'),
                    os.path.join(root, 'Imagenet64_train_part2')]
    x_data = []

    img_size2 = img_size * img_size
    tot_mean = None

    for data_folder in data_folders:
        for filename in os.listdir(data_folder):
            # read the data as numpy arrays
            with open(os.path.join(data_folder, filename), 'rb') as f:
                print("processing file: ", filename)
                print("unpickling")
                datadict = pickle.load(f, encoding='latin1')
                x = np.array(datadict['data'])
                # y = np.array(datadict['labels'])
                mean_image = datadict['mean']
                print("finished unpickling")

                print("preprocessing data")
                # performing normalization i.e. x /= np.float32(255)
                np.true_divide(x, np.float32(255), out=x, casting='unsafe')
                np.true_divide(mean_image, np.float32(255), out=mean_image, casting='unsafe')

                # Labels are indexed from 1, shift it so that indexes start at 0
                # y = [i-1 for i in y]

                np.subtract(x, mean_image, out=x, casting='unsafe')
                if tot_mean is None:
                    tot_mean = mean_image
                else:
                    tot_mean += mean_image

                x = np.dstack((x[:, :img_size2], x[:, img_size2:2*img_size2], x[:, 2*img_size2:]))
                x = x.reshape((x.shape[0], img_size, img_size, 3)).transpose(0, 3, 1, 2)

                x_data.append(x)
                # y_data.append(y)
                print("completed preprocessing\n")

    # X_train, Y_train = np.concatenate(x_data), np.concatenate(y_data)
    X_train = np.concatenate(x_data)
    tot_mean /= 10

    # print("done processing everything!! Creating mirrored versions now")
    # create mirrored images
    # X_train = x[0:data_size, :, :, :]
    # Y_train = y[0:data_size]
    # X_train_flip = X_train[:, :, :, ::-1]
    # Y_train_flip = Y_train
    # X_train = np.concatenate((X_train, X_train_flip), axis=0)
    # Y_train = np.concatenate((Y_train, Y_train_flip), axis=0)

    print("returning data now")
    return (ImageNet64Data(X_train, None),
            # CIFAR10Data(X_val, y_val),
            # CIFAR10Data(X_test, y_test),
            mean_image)


def load_data_val(root, img_size=64):
    '''
    loads the data from files and returns a torch dataset
    '''
    data_file = os.path.join(root, 'val_data')
    x_data = []

    img_size2 = img_size * img_size

    for idx in range(1, 11):
        # read the data as numpy arrays
        with open(data_file + str(idx), 'rb') as f:
            datadict = pickle.load(f, encoding='latin1')
            x = np.array(datadict['data'])
            # y = np.array(datadict['labels'])

            # performing normalization i.e. x /= np.float32(255)
            np.true_divide(x, np.float32(255), out=x, casting='unsafe')

            # Labels are indexed from 1, shift it so that indexes start at 0
            # y = [i-1 for i in y]

            x = np.dstack((x[:, :img_size2], x[:, img_size2:2*img_size2], x[:, 2*img_size2:]))
            x = x.reshape((x.shape[0], img_size, img_size, 3)).transpose(0, 3, 1, 2)

            x_data.append(x)
            # y_data.append(y)

    # X_train, Y_train = np.concatenate(x_data), np.concatenate(y_data)
    X_train = np.concatenate(x_data)

    # create mirrored images
    # X_train = x[0:data_size, :, :, :]
    # Y_train = y[0:data_size]
    # X_train_flip = X_train[:, :, :, ::-1]
    # Y_train_flip = Y_train
    # X_train = np.concatenate((X_train, X_train_flip), axis=0)
    # Y_train = np.concatenate((Y_train, Y_train_flip), axis=0)

    return (ImageNet64Data(X_train, None),
            # CIFAR10Data(X_val, y_val),
            # CIFAR10Data(X_test, y_test),
            )
