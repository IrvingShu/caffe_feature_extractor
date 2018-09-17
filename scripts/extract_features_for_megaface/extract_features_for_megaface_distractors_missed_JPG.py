#! /usr/bin/env python
import sys
from extract_features import extract_features


if __name__ == '__main__':
    config_json = './extractor_config_sphere64_pod.json'
    save_dir = 'megaface-features-sphereface-64'
    gpu_id = None

    if len(sys.argv) > 1:
        gpu_id = int(sys.argv[1])
    print 'gpu_id:', gpu_id

    # image path: osp.join(image_dir, <each line in image_list_file>)
    image_dir = r'/workspace/code/mtcnn-caffe-zyf/scripts/face_aligner/megaface_mtcnn_aligned/aligned_imgs'
    #image_list_file = r'/workspace/data/__face_datasets__/MegaFace/MegaFace_dataset/megaface-image-list-all.txt'
    #image_list_file = r'/workspace/code/caffe_feature_extractor_zyf/scripts/extract_features_for_megaface/megaface-features-sphereface-64/skipped_image_list_1_JPG.txt'
    image_list_file = r'/workspace/code/caffe_feature_extractor_zyf/scripts/extract_features_for_megaface/megaface-features-sphereface-64/skipped_image_list_2.txt'
    extract_features(config_json, save_dir, image_list_file, image_dir, gpu_id=gpu_id)
