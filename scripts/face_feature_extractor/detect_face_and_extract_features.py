#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import os.path as osp
import numpy as np
import cv2

import json
import time
import argparse
import init_paths

from face_aligner import FaceAligner
from face_detector import MtcnnDetector, draw_faces
from caffe_feature_extractor import CaffeFeatureExtractor


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('img_list_file', type=str,
                        default='./list_img.txt',
                        help='Path to image list file')
    parser.add_argument('--config', type=str,
                        default='./config_cv2.json',
                        help='Path to config file')
    parser.add_argument('--save_dir', type=str,
                        default='./feat_rlt',
                        help='Path to image list file')
    parser.add_argument('--image_root_dir', type=str,
                        default='',
                        help='root dir of image list')
    parser.add_argument('--no_detect', action='store_true',
                        help='do not detect faces if input images are cropped faces')
    parser.add_argument('--no_align', action='store_true',
                        help='do not align faces if input images are cropped faces'
                        ' and faces are aligned or do not need alignment')
    parser.add_argument('--show_image', action='store_true',
                        help='draw face rects and show images')
    parser.add_argument('--save_image', action='store_true',
                        help='save images with face rects')

    return parser.parse_args(argv)


def load_config(config_file):
    fp = open(config_file, 'r')
    config = json.load(fp)
    return config


def main(argv):
    args = parse_arguments(argv)
    print '===> args:\n', args

    config = load_config(args.config)
    print '===> config:\n', config

    extractor_config = config['face_feature']
    mtcnn_model_path = str(config['mtcnn_model_dir'])

    do_detect = not args.no_detect
    do_align = not args.no_align

    save_dir = args.save_dir
    if not osp.exists(save_dir):
        os.makedirs(save_dir)

    save_img = args.save_image
    show_img = args.show_image

    detector = None
    aligner = None

    if do_detect:
        detector = MtcnnDetector(mtcnn_model_path)

    if do_align:
        if not do_detect:
            aligner = FaceAligner(mtcnn_model_path)
        else:
            aligner = FaceAligner(None)
    else:
        aligner = None

    feature_extractor = CaffeFeatureExtractor(extractor_config)
    feat_layer = feature_extractor.get_feature_layers()[0]

    fp = open(args.img_list_file, 'r')
    fp_rlt = open(osp.join(save_dir, 'face_feature.json'), 'w')
    fp_rlt.write('[\n')
    write_comma_flag = False

    # result_list = []
    img_cnt = 0
    faces_cnt = 0
    ttl_det_time = 0.0
    ttl_feat_time = 0.0

    for line in fp:
        img_path = line.strip()
        print("\n===>" + img_path)
        if img_path == '':
            print 'empty line, not a file name, skip to next'
            continue

        if img_path[0] == '#':
            print 'skip line starts with #, skip to next'
            continue

        # result_list.append(rlt)
        if write_comma_flag:
            fp_rlt.write(',\n')
        else:
            write_comma_flag = True

        rlt = {}
        rlt["filename"] = img_path
        rlt["faces"] = []
        rlt['face_count'] = 0

        try:
            if args.image_root_dir:
                img = cv2.imread(osp.join(args.image_root_dir, img_path))
            else:
                img = cv2.imread(img_path)
            print '\n---> img.shape: ', img.shape
        except:
            print('failed to load image: ' + img_path)
            #rlt["message"] = "failed to load"
            json_str = json.dumps(rlt, indent=2)
            fp_rlt.write(json_str)
            fp_rlt.flush()
            continue

        if img is None:
            print('failed to load image: ' + img_path)

            rlt["message"] = "failed to load"
            # result_list.append(rlt)
            json_str = json.dumps(rlt, indent=2)
            fp_rlt.write(json_str)
            fp_rlt.flush()
            continue

        img_cnt += 1
        if do_detect:
            t1 = time.clock()

            bboxes, points = detector.detect_face(img)

            t2 = time.clock()
            ttl_det_time += t2 - t1
            print("detect_face() costs %f seconds" % (t2 - t1))

        else:
            print '---> Will not do detection because of option "--no_detect"'
            shp = img.shape
            rect = [0, 0, shp[1] - 1, shp[0] - 1, 1.0]
            bboxes = [rect]
            points = [None]

        n_faces = 0
        if bboxes is not None:
            n_faces = len(bboxes)

        if n_faces > 0:
            for (box, pts) in zip(bboxes, points):
                #                box = box.tolist()
                #                pts = pts.tolist()
                tmp = {'rect': box[0:4],
                       'score': box[4],
                       'pts': pts
                       }
                rlt['faces'].append(tmp)

            rlt['face_count'] = n_faces

#        print('output bboxes: ' + str(bboxes))
#        print('output points: ' + str(points))
        # toc()

        if do_detect:
            print("\n===> Detect %d images, costs %f seconds, avg time: %f seconds" % (
                img_cnt, ttl_det_time, ttl_det_time / img_cnt))

        print "---> %d faces detected" % n_faces

        if not n_faces:
            continue

        t1 = time.clock()
        if do_align:
            if points is None or points[0] is None:
                face_chips = aligner.get_face_chips(img, bboxes, None)
            else:
                face_chips = aligner.get_face_chips(img, bboxes, points)
#            face_chips = aligner.get_face_chips(img, bboxes, None)

#            face_chips = [im.astype(np.float) for im in face_chips_ubyte]
        else:
            print '---> Will not do alignment because of option "--no_align"'
            face_chips = [img.astype(np.float)]

        features = feature_extractor.extract_features_batch(face_chips)[feat_layer]
        t2 = time.clock()
        ttl_feat_time += t2 - t1
        print("Cropping and extracting features for %d faces cost %f seconds" %
              (n_faces, t2 - t1))
        faces_cnt += n_faces

        print("\n===> Extracting features for %d faces, costs %f seconds, avg time: %f seconds" % (
            faces_cnt, ttl_feat_time, ttl_feat_time / faces_cnt))

        for i, box in enumerate(bboxes):
            # feat_file = '%s_%d_rect[%d_%d_%d_%d].npy' % (
            #     osp.basename(img_path), i, box[0], box[1], box[2], box[3])
            # feat_file = osp.join(save_dir, feat_file)
            # np.save(feat_file, features[i])

            base_name = osp.basename(img_path)

            face_fn_prefix = '%s_face_%d' % (osp.splitext(base_name)[0], i)

            feat_file = face_fn_prefix + '.npy'
            np.save(osp.join(save_dir, feat_file), features[i])

            face_chip_fn = face_fn_prefix + '.jpg'
            cv2.imwrite(osp.join(save_dir, face_chip_fn), face_chips[i])

            rlt['faces'][i]['feat'] = feat_file
            rlt['faces'][i]['face_chip'] = face_chip_fn

        rlt['message'] = 'success'
#        result_list.append(rlt)
        json_str = json.dumps(rlt, indent=2)
        fp_rlt.write(json_str)
        fp_rlt.flush()

        if save_img or show_img:
            draw_faces(img, bboxes, points)

        if save_img:
            save_name = osp.join(save_dir, osp.basename(img_path))
            cv2.imwrite(save_name, img)

        if show_img:
            cv2.imshow('img', img)

            ch = cv2.waitKey(0) & 0xFF
            if ch == 27:
                break

    #json.dump(result_list, fp_rlt, indent=4)
    fp_rlt.write('\n]\n')
    fp_rlt.close()
    fp.close()

    if show_img:
        cv2.destroyAllWindows()


if __name__ == '__main__':
    argv = []
    if len(sys.argv) < 2:
        #        img_list = './list_img_tianyan.txt'
        #        img_list = './list_img_qiniu-staff.txt'
        img_list = './list_img_renlianku.txt'
        #        img_list = r'C:\zyf\00_Ataraxia\facex\facex_cluster_test_imgs-wlc\face_chips_list.txt'
#        img_list = r'C:\zyf\github\mtcnn-caffe-good\face_aligner\face_chips\list_img.txt'
#        save_dir = './tianyan_test_pics_new_fixbug_eltavg'
#        save_dir = './renlianku_qiniu_staff_fixbug_eltavg'
        save_dir = './renlianku_fixbug_eltavg'
#        save_dir = './fxcluster_test_chips_fixbug_noflip_align'

        argv.append(img_list)
#        argv.append('--no_detect')
#        argv.append('--no_align')
#        argv.append('--show_image')
#        argv.append('--save_image')
        argv.append('--save_dir=' + save_dir)
    else:
        argv = sys.argv[1:]

    main(argv)
