import os
import os.path as osp
import numpy as np

import _init_paths
from caffe_feature_extractor import CaffeFeatureExtractor


def process_image_list(feat_extractor, img_list, 
                        image_dir=None, save_dir=None):
    ftrs = feat_extractor.extract_features_for_image_list(img_list, image_dir)
#    np.save(osp.join(save_dir, save_name), ftrs)

    # root_len = len(image_dir)
    feat_layer_names = feat_extractor.get_feature_layers()

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


def main(config_json, save_dir, image_list_file, image_dir):
    if not osp.exists(save_dir):
        os.makedirs(save_dir)

    fp = open(image_list_file, 'r')

    # init a feat_extractor
    print '\n===> init a feat_extractor'
    feat_extractor = CaffeFeatureExtractor(config_json)

    batch_size = feat_extractor.get_batch_size()

    print 'feat_extractor can process %d images in a batch' % batch_size

    img_list = []
    cnt = 0
    batch_cnt = 0

    for line in fp:
        if line.startswith('#'):
            continue

        items = line.split()
        img_list.append(items[0].strip())
        cnt += 1

        if cnt == batch_size:
            batch_cnt += 1
            print '\n===> Processing batch #%d with %d images' % (batch_cnt, cnt)

            process_image_list(feat_extractor, img_list, image_dir, save_dir)
            cnt = 0
            img_list = []

    if cnt > 0:
        batch_cnt += 1
        print '\n===> Processing batch #%d with %d images' % (batch_cnt, cnt)
        process_image_list(feat_extractor, img_list, image_dir, save_dir)

    fp.close()


if __name__ == '__main__':
    config_json = '../extractor_config_sphere64.json'
    save_dir = 'feature_rlt_sphere64_noflip'

    # image path: osp.join(image_dir, <each line in image_list_file>)
#    image_dir = r'C:\zyf\github\mtcnn-caffe-zyf\face_aligner\face_chips'
#    image_list_file = r'C:\zyf\github\lfw-evaluation-zyf\extract_face_features\face_chips\face_chips_list_2.txt'
    image_dir = r'../../test_data/face_chips'
    image_list_file = r'../../test_data/face_chips_list.txt'
    main(config_json, save_dir, image_list_file, image_dir)
