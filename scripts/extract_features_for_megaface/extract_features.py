import os
import os.path as osp
import numpy as np
import json

import _init_paths
from caffe_feature_extractor import CaffeFeatureExtractor

from matio import save_mat


def get_image_sub_dir_and_save_fn(src_fn):
    spl = osp.split(src_fn)
    base_name = spl[1]
#        sub_dir = osp.split(spl[0])[1]
    sub_dir = spl[0]
    save_fn = base_name + '_feat.bin'

#    print 'sub_dir: ', sub_dir
#    print 'save_fn: ', save_fn

    return sub_dir, save_fn


def exist_src_img(img_dir, img_fn):
    return osp.exists(
        osp.join(img_dir, img_fn)
    )


def exist_dst_feats(save_dir, src_fn, layer_list=None):
    sub_dir, save_name = get_image_sub_dir_and_save_fn(src_fn)

    if isinstance(layer_list, list):
        for layer in layer_list:
            if not osp.exists(
                osp.join(save_dir, layer, sub_dir, save_name)
            ):
                return False
        return True

    elif layer_list is None:
        return osp.exists(
            osp.join(save_dir, sub_dir, save_name)
        )

    elif isinstance(layer_list, str):
        return osp.exists(
            osp.join(save_dir, layer_list, sub_dir, save_name)
        )

    else:
        return False


def process_image_list(feat_extractor, img_list,
                       image_dir=None, save_dir=None):
    ftrs = feat_extractor.extract_features_for_image_list(img_list, image_dir)
#    np.save(osp.join(save_dir, save_name), ftrs)

    # root_len = len(image_dir)
    feat_layer_names = feat_extractor.get_feature_layers()

    for i in range(len(img_list)):
        #         spl = osp.split(img_list[i])
        #         base_name = spl[1]
        # #        sub_dir = osp.split(spl[0])[1]
        #         sub_dir = spl[0]
        sub_dir, save_name = get_image_sub_dir_and_save_fn(img_list[i])

        for layer in feat_layer_names:
            #            if sub_dir:
            #                save_sub_dir = osp.join(save_dir, layer, sub_dir)
            #            else:
            #                save_sub_dir = osp.join(save_dir, layer)
            save_sub_dir = osp.join(save_dir, layer, sub_dir)

            if not osp.exists(save_sub_dir):
                os.makedirs(save_sub_dir)

            # save_name = osp.splitext(base_name)[0] + '_feat.bin'
            # np.save(osp.join(save_sub_dir, save_name), ftrs[layer][i])
            save_mat(osp.join(save_sub_dir, save_name), ftrs[layer][i])


def extract_features(config_json, save_dir,
                     image_list_file, image_dir,
                     check_src_exist=True,
                     skip_dst_exist=True,
                     gpu_id=None):

    if not osp.exists(save_dir):
        os.makedirs(save_dir)

    if gpu_id is not None:
        gpu_id = int(gpu_id)
        fp = open(config_json)
        config_json = json.load(fp)
        fp.close()

        print '===> overwirte gpu_id from {} in config file into {}'.format(
            config_json['gpu_id'], gpu_id)
        config_json['gpu_id'] = gpu_id

    fp = open(image_list_file, 'r')

    # init a feat_extractor
    print '\n===> init a feat_extractor'
    feat_extractor = CaffeFeatureExtractor(config_json)

    batch_size = feat_extractor.get_batch_size()
    print 'feat_extractor can process %d images in a batch' % batch_size

    feat_layer_names = feat_extractor.get_feature_layers()
    print 'feat_extractor will extract features from layers:', feat_layer_names

    img_list = []
    cnt = 0
    batch_cnt = 0

    fn_skip = osp.join(save_dir, 'skipped_image_list.txt')
    fp_skip = open(fn_skip, 'w')

    for line in fp:
        if line.startswith('#'):
            continue

        # items = line.split()
        # img_fn = items[0].strip()
        img_fn = line.strip()

        if check_src_exist and not exist_src_img(image_dir, img_fn):
            print '---> Skip {}, source image not found'.format(img_fn)
            fp_skip.write('{}, no source\n'.format(img_fn))
            continue

        if skip_dst_exist and exist_dst_feats(save_dir, img_fn, feat_layer_names):
            print '---> Skip {}, dst features already exist'.format(img_fn)
            fp_skip.write('{}, exist dst\n'.format(img_fn))
            continue

        img_list.append(img_fn)
        cnt += 1

        if cnt == batch_size:
            batch_cnt += 1
            print '\n===> Processing batch #%d with %d images' % (batch_cnt, cnt)

            process_image_list(feat_extractor, img_list, image_dir, save_dir)
            cnt = 0
            img_list = []

            fp_skip.flush()

    if cnt > 0:
        batch_cnt += 1
        print '\n===> Processing batch #%d with %d images' % (batch_cnt, cnt)
        process_image_list(feat_extractor, img_list, image_dir, save_dir)

    fp.close()
    fp_skip.close()


if __name__ == '__main__':
    config_json = '../extractor_config_sphere64.json'
    save_dir = 'rlt_feats_face_chips'
    gpu_id = None

    # image path: osp.join(image_dir, <each line in image_list_file>)
    # image_dir = r'C:\zyf\github\mtcnn-caffe-zyf\face_aligner\face_chips'
    # image_list_file = r'C:\zyf\github\lfw-evaluation-zyf\extract_face_features\face_chips\face_chips_list_2.txt'
    image_dir = r'../../test_data/face_chips'
    image_list_file = r'../../test_data/face_chips_list.txt'
    extract_features(config_json, save_dir, image_list_file, image_dir, gpu_id=gpu_id)
