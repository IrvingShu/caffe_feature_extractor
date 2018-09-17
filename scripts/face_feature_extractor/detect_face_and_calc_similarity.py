#! /usr/bin/env python

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
from face_detector import MtcnnDetector, draw_faces, cv2_put_text_to_image
from caffe_feature_extractor import CaffeFeatureExtractor

from numpy.linalg import norm


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('img_list_file', type=str,
                        default='./list_img.txt',
                        help='Path to image list file')
    parser.add_argument('--config', type=str,
                        default='./config_cv2.json',
                        help='Path to config file')
    parser.add_argument('--save_dir', type=str,
                        default='./results',
                        help='Path to image list file')
    parser.add_argument('--image_root_dir', type=str,
                        default='',
                        help='root dir of image list')
    parser.add_argument('--no_detect', action='store_true',
                        help='do not detect faces if input images are cropped faces')
    parser.add_argument('--no_align', action='store_true',
                        help='do not align faces if input images are pre-aligned faces, this option will set "--no_detect"')
    parser.add_argument('--show_image', action='store_true',
                        help='draw face rects and show images')
    parser.add_argument('--save_image', action='store_true',
                        help='save images with face rects')

    return parser.parse_args(argv)


def load_config(config_file):
    fp = open(config_file, 'r')
    config = json.load(fp)
    return config


def get_image_path(line, root_dir=None):
    img_path = line.strip()
    print("\n===>" + img_path)

    if img_path == '':
        print 'empty line, not a file name, skip to next'
        img_path = None

    if img_path[0] == '#':
        print 'skip line starts with #, skip to next'
        img_path = None

    if img_path and root_dir:
        img_path = osp.join(root_dir, img_path)

    return img_path


def detect_faces_and_extract_features(img_path, ctx_static, ctx_active):
    detector = ctx_static['detector']
    aligner = ctx_static['aligner']
    feature_extractor = ctx_static['feature_extractor']
    do_detect = ctx_static['do_detect']
    do_align = ctx_static['do_align']
    save_img = ctx_static['save_img']
    show_img = ctx_static['show_img']
    save_dir = ctx_static['save_dir']
    max_faces = ctx_static['max_faces']

    img_cnt = ctx_active['img_cnt']
    faces_cnt = ctx_active['faces_cnt']
    ttl_det_time = ctx_active['ttl_det_time']
    ttl_feat_time = ctx_active['ttl_feat_time']

    rlt = {}
    rlt["filename"] = img_path
    rlt["faces"] = []
    rlt['face_count'] = 0

    try:
        img = cv2.imread(img_path)
    except:
        print('failed to load image: ' + img_path)
        rlt["message"] = "failed to load"

        ctx_active['img_cnt'] = img_cnt
        ctx_active['faces_cnt'] = faces_cnt
        ctx_active['ttl_det_time'] = ttl_det_time
        ctx_active['ttl_feat_time'] = ttl_feat_time

        return rlt, None

    if img is None:
        print('failed to load image: ' + img_path)

        rlt["message"] = "failed to load"

        ctx_active['img_cnt'] = img_cnt
        ctx_active['faces_cnt'] = faces_cnt
        ctx_active['ttl_det_time'] = ttl_det_time
        ctx_active['ttl_feat_time'] = ttl_feat_time

        return rlt, None

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

    if bboxes is not None and len(bboxes) > 0:
        for (box, pts) in zip(bboxes, points):
            #                box = box.tolist()
            #                pts = pts.tolist()
            tmp = {'rect': box[0:4],
                   'score': box[4],
                   'pts': pts
                   }
            rlt['faces'].append(tmp)

        rlt['face_count'] = len(bboxes)

    print("\n===> Detect %d images, costs %f seconds, avg time: %f seconds" % (
        img_cnt, ttl_det_time, ttl_det_time / img_cnt))

    if bboxes is None:
        ctx_active['img_cnt'] = img_cnt
        ctx_active['faces_cnt'] = faces_cnt
        ctx_active['ttl_det_time'] = ttl_det_time
        ctx_active['ttl_feat_time'] = ttl_feat_time

        return rlt, None

    if len(bboxes) > max_faces:
        print 'len(bboxes) > max_faces=%d, Only keep first %d faces' % (max_faces, max_faces)
        bboxes = bboxes[:max_faces]
        points = points[:max_faces]

    t1 = time.clock()
    if do_align:
        if points is None or points[0] is None:
            face_chips = aligner.get_face_chips(img, bboxes, None)
        else:
            face_chips = aligner.get_face_chips(img, bboxes, points)
#        face_chips_ubyte = aligner.get_face_chips(img, bboxes, None)

#        face_chips = [im.astype(np.float) for im in face_chips_ubyte]
    else:
        print '---> Will not do alignment because of option "--no_align"'
#        face_chips = [img.astype(np.float)]
        face_chips = [img]

    feat_layer = feature_extractor.get_feature_layers()[0]

#    face_chips = aligner.get_face_chips(img, bboxes, points)
#    imgs = [chip.astype(np.float) for chip in face_chips]
    features = feature_extractor.extract_features_batch(face_chips)[feat_layer]
    t2 = time.clock()
    ttl_feat_time += t2 - t1
    print("Cropping and extracting features for %d faces cost %f seconds" %
          (len(bboxes), t2 - t1))
    faces_cnt += len(bboxes)

    print("\n===> Extracting features for %d faces, costs %f seconds, avg time: %f seconds" % (
        faces_cnt, ttl_feat_time, ttl_feat_time / faces_cnt))

    for i, box in enumerate(bboxes):
        # feat_file = '%s_%d_rect[%d_%d_%d_%d].npy' % (
        #     osp.basename(img_path), i, box[0], box[1], box[2], box[3])
        # feat_file = osp.join(save_dir, feat_file)

        # np.save(feat_file, features[i])

        # rlt['faces'][i]['feat'] = feat_file
        base_name = osp.basename(img_path)

        face_fn_prefix = '%s_face_%d' % (osp.splitext(base_name)[0], i)

        feat_file = face_fn_prefix + '.npy'
        np.save(osp.join(save_dir, feat_file), features[i])

        face_chip_fn = face_fn_prefix + '.jpg'
        cv2.imwrite(osp.join(save_dir, face_chip_fn), face_chips[i])

        rlt['faces'][i]['feat'] = feat_file
        rlt['faces'][i]['face_chip'] = face_chip_fn

    rlt['message'] = 'success'

    if save_img or show_img:
        draw_faces(img, bboxes, points)

    if save_img:
        save_name = osp.join(save_dir, osp.basename(img_path))
        cv2.imwrite(save_name, img)

    if show_img:
        cv2.imshow('img', img)

        ch = cv2.waitKey(0) & 0xFF
#            if ch == 27:
#                break

    ctx_active['img_cnt'] = img_cnt
    ctx_active['faces_cnt'] = faces_cnt
    ctx_active['ttl_det_time'] = ttl_det_time
    ctx_active['ttl_feat_time'] = ttl_feat_time

    return rlt, features, face_chips


def calc_similarity(feat1, feat2):
    feat1_norm = norm(feat1)
    feat2_norm = norm(feat2)

    print 'feat1_norm:', feat1_norm
    print 'feat2_norm:', feat2_norm

    sim = np.dot(feat1, feat2) / (feat1_norm * feat2_norm)

    print 'sim:', sim

    return sim


def main(argv):
    args = parse_arguments(argv)
    print '===> args:\n', args

    config = load_config(args.config)
    print '===> config:\n', config

    max_faces = config['max_faces']
    extractor_config = config['face_feature']
    mtcnn_model_path = str(config['mtcnn_model_dir'])

    do_detect = not args.no_detect
    do_align = not args.no_align

    save_dir = args.save_dir
    if not osp.exists(save_dir):
        os.makedirs(save_dir)

    pair_save_dir = osp.join(save_dir, 'img_pairs')
    if not osp.exists(pair_save_dir):
        os.mkdir(pair_save_dir)

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

    ctx_static = {}
    #ctx_static['args'] = args
    ctx_static['detector'] = detector
    ctx_static['aligner'] = aligner
    ctx_static['feature_extractor'] = feature_extractor
    ctx_static['do_detect'] = do_detect
    ctx_static['do_align'] = do_align
    ctx_static['save_img'] = save_img
    ctx_static['show_img'] = show_img
    ctx_static['save_dir'] = save_dir
    ctx_static['max_faces'] = max_faces

#    result_list = []
    img_cnt = 0
    faces_cnt = 0
    ttl_det_time = 0.0
    ttl_feat_time = 0.0

    ctx_active = {}
    #ctx_active['result_list'] = result_list
    ctx_active['img_cnt'] = img_cnt
    ctx_active['faces_cnt'] = faces_cnt
    ctx_active['ttl_det_time'] = ttl_det_time
    ctx_active['ttl_feat_time'] = ttl_feat_time

    fp = open(args.img_list_file, 'r')
    fp_rlt = open(osp.join(save_dir, 'face_feature.json'), 'w')
    fp_rlt.write('[\n')
    write_comma_flag = False

    while True:
        line = fp.readline().strip()
        print '---> line: ', line
        if not line:
            break

        img_path = get_image_path(line, args.image_root_dir)
        print '---> img_path: ', img_path

        (rlt, features, face_chips) = detect_faces_and_extract_features(
            img_path, ctx_static, ctx_active)
#        print 'features: ', features
#        print 'id(features): ', id(features)

        # result_list.append(rlt)
        if write_comma_flag:
            fp_rlt.write(',\n')
        else:
            write_comma_flag = True

        json_str = json.dumps(rlt, indent=2)
        fp_rlt.write(json_str)
        fp_rlt.flush()

        line = fp.readline().strip()
        print '---> line: ', line
        if not line:
            break

        img_path2 = get_image_path(line, args.image_root_dir)
        print '---> img_path2: ', img_path2

        (rlt2, features2, face_chips2) = detect_faces_and_extract_features(
            img_path2, ctx_static, ctx_active)
#        print 'features2: ', features2
#        print 'features: ', features
#
#        print 'id(features): ', id(features)
#        print 'id(features2): ', id(features2)
#
#        print 'features.data: ', id(features.data)
#        print 'features2.data: ', id(features2.data)

        # result_list.append(rlt2)
        json_str = json.dumps(rlt2, indent=2)
        fp_rlt.write(',\n' + json_str)
        fp_rlt.flush()

        if rlt['face_count'] and rlt2['face_count']:
            #            sim = calc_similarity(features[0], features2[0])
            #            img_pair = np.hstack((face_chips[0], face_chips2[0]))
            #            img_pair_fn = '%s_%d_vs_%s_%d_%5.4f.jpg' % (osp.basename(img_path), 0, osp.basename(img_path2), 0, sim)
            #            img_pair_fn = osp.join(pair_save_dir, img_pair_fn)
            #            cv2.imwrite(img_pair_fn, img_pair)
            #
            #            print '---> similarity: ', sim

            for j in range(rlt['face_count']):
                for i in range(rlt2['face_count']):
                    sim = calc_similarity(features[j], features2[i])
                    print 'features[%d]: ' % j, features[j]
                    print 'features2[%d]: ' % i, features2[i]

                    img_pair = np.hstack((face_chips[j], face_chips2[i]))

                    img_pair_fn = '%s_%d_vs_%s_%d_%5.4f.jpg' % (
                        osp.basename(img_path), j, osp.basename(img_path2), i, sim)
                    img_pair_fn = osp.join(pair_save_dir, img_pair_fn)

                    sim_txt = '%5.4f' % sim
                    cv2_put_text_to_image(
                        img_pair, sim_txt, 40, 5, 30, (0, 0, 255))
                    cv2.imwrite(img_pair_fn, img_pair)

                    print '---> similarity: ', sim

    # json.dump(result_list, fp_rlt, indent=2)
    fp_rlt.write('\n]\n')

    fp_rlt.close()
    fp.close()

    if show_img:
        cv2.destroyAllWindows()


if __name__ == '__main__':
    argv = []
    if len(sys.argv) < 2:
        img_list = './list_img_tianyan_10.txt'

        argv.append(img_list)
    #    argv.append('--no_detect')
#        argv.append('--no_align')
#        argv.append('--show_image')
        argv.append('--save_image')
    else:
        argv = sys.argv[1:]

    main(argv)
