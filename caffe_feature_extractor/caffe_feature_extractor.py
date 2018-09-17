#!/bin/usr/env python

# by zhaoyafei0210@gmail.com

import os
import os.path as osp

import numpy as np
from numpy.linalg import norm
# import scipy.io as sio
import skimage

import json
import time

import _init_paths

try:
    import caffe
except ImportError as err:
    raise ImportError('{}. Please set the correct caffe_root in {} '
                      'or in the first line of your main python script.'.format(
                          err, osp.abspath(osp.dirname(__file__)) + '/_init_paths.py')
                      )


# from caffe import Classifier
from classifier import Classifier

# from collections import OrderedDict


def load_binaryproto(bp_file):
    blob_proto = caffe.proto.caffe_pb2.BlobProto()
    data = open(bp_file, 'rb').read()
    blob_proto.ParseFromString(data)
    arr = caffe.io.blobproto_to_array(blob_proto)
#    # printtype(arr)
    return arr


class Error(Exception):
    """Base class for exceptions in this module."""
    pass


class InitError(Error):
    """ Class for Init exceptions in this module."""
    pass


class FeatureLayerError(Error):
    """Exception for Invalid feature layer names."""
    pass


class ExtractionError(Error):
    """Exception from extract_xxx()."""
    pass


class CaffeFeatureExtractor(object):
    def __init__(self, config_json):
        self.net = None
#        self.net_blobs = None
        self.input_shape = None
        self.batch_size = None

        self.config = {
            #'network_prototxt': '/path/to/prototxt',
            #'network_caffemodel': '/path/to/caffemodel',
            #'data_mean': '/path/to/the/mean/file',
            #'feature_layer': 'fc5',
            'batch_size': 1,
            'input_scale': 1.0,
            'raw_scale': 1.0,
            # default is BGR, be careful of your input image's channel
            'channel_swap': (2, 1, 0),
            # 0,None - will not use mirror_trick, 1 - eltavg (i.e.
            # eltsum()*0.5), 2 - eltmax
            'mirror_trick': 0,
            'image_as_grey': False,
            'normalize_output': False,
            'cpu_only': 0,
            'gpu_id': 0
        }

        if isinstance(config_json, str):
            if osp.isfile(config_json):
                fp = open(config_json, 'r')
                _config = json.load(fp)
                fp.close()
            else:
                _config = json.loads(config_json)
        else:
            _config = config_json

        # must convert to str, because json.load() outputs unicode which is not support
        # in caffe's cpp function
        _config['network_prototxt'] = str(_config['network_prototxt'])
        _config['network_caffemodel'] = str(_config['network_caffemodel'])
        _config['data_mean'] = str(_config['data_mean'])
        _config['feature_layer'] = str(_config['feature_layer'])
        _config['channel_swap'] = tuple(
            [int(i.strip()) for i in _config['channel_swap'].split(',')])

        self.config.update(_config)

        mean_arr = None
        if (self.config['data_mean']):
            try:
                if self.config['data_mean'].endswith('.npy'):
                    mean_arr = np.load(self.config['data_mean'])
                elif self.config['data_mean'].endswith('.binaryproto'):
                    mean_arr = load_binaryproto(self.config['data_mean'])
                else:
                    mean_arr = np.matrix(self.config['data_mean']).A1
                # # print'mean array shape: ', mean_arr.shape
                # # print'mean array: \n', mean_arr
            except:
                raise InitError('Failed to load "data_mean": ' +
                                str(self.config['data_mean']))

        if (int(self.config['mirror_trick']) not in [0, 1, 2]):
            raise InitError('"mirror_trick" must be one from [0,1,2]')

        # print'\n===> CaffeFeatureExtractor.config: \n', self.config

        try:
            if(self.config['cpu_only']):
                caffe.set_mode_cpu()
            else:
                caffe.set_mode_gpu()
                caffe.set_device(int(self.config['gpu_id']))
        except Exception as err:
            raise InitError(
                'Exception from caffe.set_mode_xxx() or caffe.set_device(): ' + str(err))

        try:
            self.net = Classifier(self.config['network_prototxt'],
                                  self.config['network_caffemodel'],
                                  None,
                                  mean_arr,
                                  self.config['input_scale'],
                                  self.config['raw_scale'],
                                  self.config['channel_swap']
                                  )
        except Exception as err:
            raise InitError('Exception from Clssifier.__init__(): ' + str(err))

        # if (self.config['feature_layer'] not in self.net.layer_dict.keys()):
        #     raise FeatureLayerError('Invalid feature layer names: '
        #                             + self.config['feature_layer'])
        self.config['feature_layer'] = self.get_feature_layers(
            self.config['feature_layer'])

#        self.net_blobs = OrderedDict([(k, v.data)
#                                  for k, v in self.net.blobs.items()])
#        # print'self.net_blobs: ', self.net_blobs
#        for k, v in self.net.blobs.items():
#            # printk, v

        self.input_shape = self.net.blobs['data'].data.shape
        # print'---> original input data shape (in prototxt): ', self.input_shape
        # print'---> original batch_size (in prototxt): ', self.input_shape[0]

        self.batch_size = self.config['batch_size']
        # print'---> batch size in the config: ', self.batch_size

        if self.config['mirror_trick'] > 0:
            # print'---> need to double the batch size of the net input data
            # because of mirror_trick'
            final_batch_size = self.batch_size * 2
        else:
            final_batch_size = self.batch_size

        # print'---> will use a batch size: ', final_batch_size

        # reshape net into final_batch_size
        if self.input_shape[0] != final_batch_size:
            try:
                # print'---> reshape net input batch size from %d to %d' %
                # (self.input_shape[0], final_batch_size)
                self.net.blobs['data'].reshape(
                    final_batch_size, self.input_shape[1], self.input_shape[2], self.input_shape[3])
                # print'---> reshape the net blobs'
                self.net.reshape()
            except Exception as err:
                raise InitError('Exception when reshaping net: ' + str(err))

            self.input_shape = self.net.blobs['data'].data.shape

            # print'---> after reshape, net input data shape: ',
            # self.input_shape

        # print'---> the final input data shape: ', self.input_shape

#        if self.config['mirror_trick'] > 0:
#            if self.batch_size < 2:
#                raise InitError('If using mirror_trick, batch_size of input "data" layer must > 1')
#
#            self.batch_size /= 2
# # print'halve the batch_size for mirror_trick eval: batch_size=',
# self.batch_size

    # def __delete__(self):
    #     print'delete CaffeFeatureExtractor object'
    #     pass

    def load_image(self, image_path, mirror=False):
        img = caffe.io.load_image(
            image_path, color=not self.config['image_as_grey'])
        if self.config['image_as_grey'] and img.shape[2] != 1:
            img = skimage.color.rgb2gray(img)
            img = img[:, :, np.newaxis]

        if mirror:
            img = np.fliplr(mirror)

        return img

    def load_images(self, image_path):
        pass

    def split_layer_names(self, layer_names):
        if isinstance(layer_names, list):
            return layer_names
        elif isinstance(layer_names, str):
            spl = layer_names.split(',')
            layers = [layer.strip() for layer in spl]
            return layers
        else:
            raise FeatureLayerError('layer_names must be '
                                    'a list of layer names, or a string with '
                                    'layer names seperated by comma.'
                                    'Input layer_names is: {}'.format(
                                        layer_names)
                                    )

    def get_feature_layers(self, layer_names=None):
        if not layer_names:
            return self.config['feature_layer']

        layer_names = self.split_layer_names(layer_names)

        for layer in layer_names:
            if (layer not in self.net.layer_dict.keys()):
                raise FeatureLayerError(
                    'Invalid feature layer name:'.format(layer)
                )

        return layer_names

    def get_layer_names(self):
        layer_names = self.net.layer_dict.keys()
        return layer_names

    def get_first_layer_name(self):
        return self.get_layer_names()[0]

    def get_final_layer_name(self):
        return self.get_layer_names()[-1]

    def get_batch_size(self):
        return self.batch_size

    def set_feature_layers(self, layer_names):
        layer_names = self.get_feature_layers(layer_names)
        self.config['feature_layer'] = layer_names

    def extract_feature(self, image, layer_names=None, mirror_input=False):
        #        layer_names = self.get_feature_layers(layer_names)
        #
        #        for layer in layer_names:
        #            feat_shp = self.net.blobs[layer].data.shape
        #            # print'layer "{}"feature shape: {}'.format(layer, feat_shp)
        #
        #        img_batch = []
        #        cnt_load_img = 0
        #        cnt_predict = 0
        #
        #        time_load_img = 0.0
        #        time_predict = 0.0
        # print'---> Calling extract_feature():'

        if isinstance(image, str):
            #            t1 = time.clock()
            img = self.load_image(image)
#            cnt_load_img += 1
#            t2 = time.clock()
#            time_predict += (t2 - t1)
        else:
            img = image.astype(np.float32)  # data type must be float32

        # print'image shape: ', img.shape

#        img_batch.append(img)
#
#        if self.config['mirror_trick']:
#            mirror_img = np.fliplr(img)
#            img_batch.append(mirror_img)
#            # print'add mirrored images into predict batch'
#            # print'after add: len(img_batch)=%d' % (len(img_batch))
#
#        n_imgs = 1
#        t1 = time.clock()
#
#        self.net.predict(img_batch, oversample=False)
#
#        t2 = time.clock()
#        time_predict += (t2 - t1)
#        cnt_predict += n_imgs
#
#        features_dict = {}
#        for layer in layer_names:
#            # must call blobs_data(v) again, because it invokes (mutable_)
#            # cpu_data() which syncs the memory between GPU and CPU
#            #        blobs = OrderedDict([(k, v.data)
#            #                             for k, v in self.net.blobs.items()])
#            #        # print'blobs: ', blobs
#            feat_blob_data = self.net.blobs[layer].data
#
#            if self.config['mirror_trick']:
#                #            ftrs = blobs[layer_names][0:n_imgs * 2, ...]
#                ftrs = feat_blob_data[0:n_imgs * 2, ...]
#                if self.config['mirror_trick'] == 2:
#                    eltop_ftrs = np.maximum(
#                        ftrs[:n_imgs], ftrs[n_imgs:n_imgs * 2])
#                else:
#                    eltop_ftrs = (ftrs[:n_imgs] +
#                                  ftrs[n_imgs::n_imgs * 2]) * 0.5
#
#                feature = eltop_ftrs[0]
#
#            else:
#                #            ftrs = blobs[layer_names][0:n_imgs, ...]
#                ftrs = feat_blob_data[0:n_imgs, ...]
#                feature = ftrs.copy()  # copy() is a must-have
#
#            if cnt_load_img:
#                # print('load %d images cost %f seconds, average time: %f seconds'
#                       % (cnt_load_img, time_load_img, time_load_img / cnt_load_img))
#
#            # print('predict %d images cost %f seconds, average time: %f seconds'
#                   % (cnt_predict, time_predict, time_predict / cnt_predict))
#
#            feature = np.asarray(feature, dtype='float32')
#
#            if self.config['normalize_output']:
#                feat_norm = norm(feature)
#
#                # inplace-operation
#                feature /= feat_norm
#
#            features_dict[layer] = feature
        features_dict = self.extract_features_batch(
            [img], layer_names, mirror_input)
        return features_dict

    def extract_features_batch(self, images, layer_names=None, mirror_input=False):
        layer_names = self.get_feature_layers(layer_names)

        n_imgs = len(images)

        # print'---> Calling extract_features_batch():'
        if (n_imgs > self.batch_size
                or (self.config['mirror_trick'] and n_imgs / 2 > self.batch_size)):
            raise ExtractionError(
                'Number of input images > batch_size set in prototxt')

        features_dict = {}
        for layer in layer_names:
            feat_shp = self.net.blobs[layer].data.shape
            # print'layer "{}"feature shape: {}'.format(layer, feat_shp)

            # features_shape = (len(images),) + feat_shp[1:]
            # features = np.empty(features_shape, dtype='float32', order='C')
            # # print'output features shape: ', features_shape

            # features_dict[layer] = features

        # data type must be float32
        img_batch = [im.astype(np.float32) for im in images]

        cnt_predict = 0
        time_predict = 0.0

        if mirror_input:
            for i in range(n_imgs):
                mirror_img = np.fliplr(img_batch[i])
                img_batch[i] = mirror_img

        if self.config['mirror_trick'] > 0:
            for i in range(n_imgs):
                mirror_img = np.fliplr(img_batch[i])
                img_batch.append(mirror_img)
            # print'add mirrored images into predict batch'
            # print'after add: len(img_batch)=%d' % (len(img_batch))

        t1 = time.clock()

        self.net.predict(img_batch, oversample=False)

        t2 = time.clock()
        time_predict += (t2 - t1)
        cnt_predict += n_imgs

        for layer in layer_names:
            # must call blobs_data(v) again, because it invokes (mutable_)cpu_data() which
            # syncs the memory between GPU and CPU
            #        blobs = OrderedDict([(k, v.data)
            #                             for k, v in self.net.blobs.items()])
            #        # print'blobs: ', blobs
            feat_blob_data = self.net.blobs[layer].data

            if self.config['mirror_trick']:
                #            ftrs = blobs[layer_names][0:n_imgs * 2, ...]
                ftrs = feat_blob_data[0:n_imgs * 2, ...]
                if self.config['mirror_trick'] == 2:
                    eltop_ftrs = np.maximum(
                        ftrs[:n_imgs], ftrs[n_imgs:n_imgs * 2])
                else:
                    eltop_ftrs = (ftrs[:n_imgs] +
                                  ftrs[n_imgs:n_imgs * 2]) * 0.5

                features_dict[layer] = eltop_ftrs.copy()

            else:
                #            ftrs = blobs[layer_names][0:n_imgs, ...]
                ftrs = feat_blob_data[0:n_imgs, ...]
                features_dict[layer] = ftrs.copy()  # copy() is a must-have

            # print('Predict %d images, cost %f seconds, average time: %f seconds' %
            #      (cnt_predict, time_predict, time_predict / cnt_predict))

            # features = np.asarray(features, dtype='float32')
            if self.config['normalize_output']:
                feat_norm = norm(features_dict[layer], axis=1)

                # use inplace-operation
                features_dict[layer] /= np.reshape(feat_norm, [-1, 1])

        return features_dict

    def extract_features_for_image_list(self, image_list, img_root_dir=None,
                                        layer_names=None, mirror_input=False):
        layer_names = self.get_feature_layers(layer_names)

        features_dict = {}

        # print'---> Calling extract_features_for_image_list():'

        for layer in layer_names:
            feat_shp = self.net.blobs[layer].data.shape
            # print'layer "{}"feature shape: {}'.format(layer, feat_shp)

            features_shape = (len(image_list),) + feat_shp[1:]
            features = np.empty(features_shape, dtype='float32', order='C')
            # print'output features shape: ', features_shape
            features_dict[layer] = features

        # feat_shp = self.net.blobs[layer_names].data.shape
        # # print'feature layer shape: ', feat_shp

        # features_shape = (len(image_list),) + feat_shp[1:]
        # features = np.empty(features_shape, dtype='float32', order='C')
        # # print'output features shape: ', features_shape

        img_batch = []

        cnt_load_img = 0
        time_load_img = 0.0
#        cnt_predict = 0
#        time_predict = 0.0

        for cnt, path in zip(range(features_shape[0]), image_list):
            t1 = time.clock()

            if img_root_dir:
                path = osp.join(img_root_dir, path)

            img = self.load_image(path)
            # if cnt == 0:
            # print'image shape: ', img.shape

            img_batch.append(img)
            t2 = time.clock()

            cnt_load_img += 1
            time_load_img += (t2 - t1)

            # # print'image shape: ', img.shape
            # # printpath, type(img), img.mean()
            if (len(img_batch) == self.batch_size) or cnt == features_shape[0] - 1:
                n_imgs = len(img_batch)
                layer_ftrs_dict = self.extract_features_batch(
                    img_batch, layer_names, mirror_input)

                for layer in layer_names:
                    features_dict[layer][cnt - n_imgs +
                                         1:cnt + 1, ...] = layer_ftrs_dict[layer]

                img_batch = []

        # print('Load %d images, cost %f seconds, average time: %f seconds' %
        #       (cnt_load_img, time_load_img, time_load_img / cnt_load_img))

        return features_dict


if __name__ == '__main__':
    def load_image_list(list_file_name):
        # list_file_path = os.path.join(img_dir, list_file_name)
        f = open(list_file_name, 'r')
        img_fn_list = []

        for line in f:
            if line.startswith('#'):
                continue

            items = line.split()
            img_fn_list.append(items[0].strip())

        f.close()

        return img_fn_list

    config_json = './extractor_config.json'
    save_dir = 'feature_rlt_sphere64_eltavg_norm'

    image_dir = r'../test_data/face_chips'
    image_list_file = r'../test_data/face_chips_list.txt'

    if not osp.exists(save_dir):
        os.makedirs(save_dir)

    # test extract_features_for_image_list()
    save_name = 'img_list_features.npy'

    img_list = load_image_list(image_list_file)

    # init a feat_extractor, use a context to release caffe objects
    print '\n===> init a feat_extractor'
    feat_extractor = CaffeFeatureExtractor(config_json)
    feat_layer_names = feat_extractor.get_feature_layers()

    # print'\n===> test extract_features_for_image_list()'

    ftrs = feat_extractor.extract_features_for_image_list(img_list, image_dir)
#    np.save(osp.join(save_dir, save_name), ftrs)

    root_len = len(image_dir)

    for i in range(len(img_list)):
        spl = osp.split(img_list[i])
        base_name = spl[1]
#        sub_dir = osp.split(spl[0])[1]
        sub_dir = spl[0]
        for layer in feat_layer_names:
            if sub_dir:
                save_sub_dir = osp.join(save_dir, layer, sub_dir)
            else:
                save_sub_dir = osp.join(save_dir, layer)

            if not osp.exists(save_sub_dir):
                os.makedirs(save_sub_dir)

            save_name = osp.splitext(base_name)[0] + '.npy'
            np.save(osp.join(save_sub_dir, save_name), ftrs[layer][i])

    # test extract_feature()
    print '\n===> test extract_feature()'
    save_name_2 = 'single_feature.npy'
    layer = feat_layer_names[0]
    ftr = feat_extractor.extract_feature(osp.join(image_dir, img_list[0]))
    np.save(osp.join(save_dir, save_name_2), ftr[layer])

    ft_diff = ftr[layer] - ftrs[layer][0]
    print 'ft_diff: ', ft_diff.sum()
