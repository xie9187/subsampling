import matplotlib.pyplot as plt
import numpy as np
import os, sys, glob
import csv, h5py
import pptk

def load_h5(fname):
    """ load .h5 file and return a dict """
    f = h5py.File(fname, 'r')
    data = dict()
    for k in f.keys():
        data[k] = f[k][:]
    return data

def parse_datafile(f):
    """ parse date list file, e.g. train_files.txt """
    f = os.path.abspath(os.path.expanduser(f))
    root_dir = os.path.join('/',*f.split('/')[:-3]) # should be -1
    data_list = []
    for line in open(f):
        data_list.append(os.path.join(root_dir, line.rstrip()))
    return data_list

def load_data(data_file_list, n_points):
    X = []
    Y = []
    for data_file in data_file_list:
        package = load_h5(data_file)
        X.append(package['data'][:,:n_points])
        Y.append(package['label'][:,:n_points])
    X = np.concatenate(X)
    Y = np.squeeze(np.concatenate(Y))
    data = [[x, y] for x, y in zip(X, Y)]
    return data

def read_modelnet40_data(data_path, n_points=1024):
    train_file_list = [os.path.join(data_path, file) for file in os.listdir(data_path) if ('train' in file and '.h5' in file)]
    test_file_list = [os.path.join(data_path, file) for file in os.listdir(data_path) if ('test' in file and '.h5' in file)]
    class_name_file = [os.path.join(data_path, file) for file in os.listdir(data_path) if 'shape_names' in file][0]
    class_name_list = [line.rstrip('\n') for line in open(class_name_file)]
    train_data = load_data(train_file_list, n_points)
    test_data = load_data(test_file_list, n_points)
    return train_data, test_data, class_name_list

def read_mnist_data(data_path):
    with open(data_path, 'rt') as csvfile:
        data = np.stack(list(csv.reader(csvfile, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)), axis=0)
    # convert to 2d point cloud
    x = data[:, 1:].astype(np.int32)
    y = data[:, 0].astype(np.int32)
    data = []
    for image_1d, category in zip(x, y):
        image_2d = np.reshape(image_1d, [flags.num_row, flags.num_col])
        # show_img(image_2d)
        rows, columns = np.where(image_2d >= 0)
        val = image_2d[rows, columns]/255.
        points_xyv = np.stack([rows/float(flags.num_row), columns/float(flags.num_col), val], axis=1).astype(np.float32)
        # show_points_as_img(points_xyv)
        data.append([points_xyv, category])
    return data

def show_points_as_img(points_xyv):
    image_2d = np.zeros([flags.num_row, flags.num_col], dtype=np.float32)
    image_2d[(points_xyv[:, 0]*flags.num_row).astype(np.int32), 
              (points_xyv[:, 1]*flags.num_col).astype(np.int32)] = points_xyv[:, 2]
    plt.figure('point 2 image')
    plt.imshow(image_2d, cmap="gray")
    plt.pause(0.1)

def batch_point2img(sampled_points_xyv, points_xyv, score): # b, n, 3
    image_2d = np.zeros([flags.batch_size, flags.num_row, flags.num_col], dtype=np.float32)
    score_2d = np.zeros([flags.batch_size, flags.num_row, flags.num_col], dtype=np.float32)
    score = np.reshape(score, [flags.batch_size, flags.num_pt])
    for i in range(flags.batch_size):
        x = (sampled_points_xyv[i, :, 0]*flags.num_row).astype(np.int32)
        y = (sampled_points_xyv[i, :, 1]*flags.num_col).astype(np.int32)
        x[x > flags.num_row-1] = flags.num_row - 1
        y[y > flags.num_col-1] = flags.num_col - 1
        x[x < 0] = 0
        y[y < 0] = 0
        image_2d[i, x, y] = sampled_points_xyv[i, :, 2]
        score_2d[i, (points_xyv[i, :, 0]*flags.num_row).astype(np.int32), 
                    (points_xyv[i, :, 1]*flags.num_col).astype(np.int32)] = score[i, :]
    return np.reshape(image_2d, [flags.batch_size, flags.num_row, flags.num_col, 1]), \
           np.reshape(score_2d, [flags.batch_size, flags.num_row, flags.num_col, 1])


def show_img(image_2d):
    plt.figure('raw image')
    plt.imshow(image_2d, cmap="gray")
    plt.pause(0.1)

def get_a_batch(data, start, batch_size):
    batch_x = []
    batch_y = []
    for i in range(batch_size):
        sample = data[min(start+i, len(data)-1)]
        points = sample[0]
        np.random.shuffle(points)
        # show_points_as_img(points)
        batch_x.append(points)
        batch_y.append(sample[1])
    return np.stack(batch_x), np.stack(batch_y)

if __name__ == '__main__':
    data_path = "C:\Work\data\modelnet40_ply_hdf5_2048"
    train_data, test_data, class_name_list = read_modelnet40_data(data_path)
    v = pptk.viewer(train_data[0][0])
    print(class_name_list[train_data[0][1]])