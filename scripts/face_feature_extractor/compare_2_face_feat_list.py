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

from face_detector.mtcnn_detector import draw_faces, cv2_put_text_to_image

from numpy.linalg import norm


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('feat_list1_file', type=str,
                        default='./feat_list1.txt',
                        help='Path to the first face feature list file')
    parser.add_argument('feat_list2_file', type=str,
                        default='./feat_list2.txt',
                        help='Path to the second face feature list file')
    parser.add_argument('--save_dir', type=str,
                        default='./2_feat_list_compare_rlt',
                        help='Path to image list file')
    parser.add_argument('--root_dir1', type=str,
                        default='',
                        help='root dir of feat files in list1')
    parser.add_argument('--root_dir2', type=str,
                        default='',
                        help='root dir of feat files in list1')

    return parser.parse_args(argv)


def load_feature(feat_npy, face_img_fn=None, root_dir=None):
    if root_dir:
        feat_npy = osp.join(root_dir, feat_npy)

    if not face_img_fn:
        base_name = osp.splitext(feat_npy)[0]
        face_img_fn = base_name + '.jpg'

    feat = None
    face_img = None

    if osp.exists(feat_npy):
        feat = np.load(feat_npy)

        if osp.exists(face_img_fn):
            face_img = cv2.imread(face_img_fn)
        else:
            print "Counld not find face image: ", face_img_fn
    else:
        print "Counld not find face feature file: ", feat_npy

    return (feat, face_img)


def load_feature_list(feat_list_file, root_dir=None):
    fp = open(feat_list_file, 'r')
    feat_list = []
    face_img_list = []
    for line in fp:
        feat, face_img = load_feature(line, root_dir=root_dir)
        if feat:
            feat_list.append(feat)
            face_img_list.append(face_img)

    fp.close()

    return (feat_list, face_img_list)


def load_feature_json(feat_json_file, root_dir=None):
    fp = open(feat_json_file, 'r')
    face_dict_list = json.load(fp)
    fp.close()

    feat_list = []
    face_img_list = []
    for face_dict in face_dict_list:
        for face in face_dict['faces']:
            feat, face_img = load_feature(
                face['feat'], face['face_chip'], root_dir)

            if feat:
                feat_list.append(feat)
                face_img_list.append(face_img)

    fp.close()

    return (feat_list, face_img_list)


def get_feat_fn_list_from_json(feat_json_file):
    fp = open(feat_json_file, 'r')
    face_dict_list = json.load(fp)
    fp.close()

    feat_fn_list = []
    face_img_fn_list = []
    for face_dict in face_dict_list:
        for face in face_dict['faces']:
            feat_fn_list.append(face['feat'])
            face_img_fn_list.append('face_chip')

    fp.close()

    return (feat_fn_list, face_img_fn_list)


def get_feat_fn_list(feat_list_file):
    if feat_list_file.endswith('.json'):
        feat_fn_list, face_img_fn_list = get_feat_fn_list_from_json(
            feat_list_file)
    else:
        fp = open(feat_list_file, 'r')
        feat_fn_list = [line.strip() for line in fp]
        fp.close()

        face_img_fn_list = None

    return feat_fn_list, face_img_fn_list


def calc_similarity(feat1, feat2):
    feat1_norm = norm(feat1)
    feat2_norm = norm(feat2)

    print 'feat1_norm:', feat1_norm
    print 'feat2_norm:', feat2_norm

    sim = np.dot(feat1, feat2) / (feat1_norm * feat2_norm)

    print 'sim:', sim

    return sim


def calc_similarity_L2(feat1, feat2):
    diff_vec = feat1 - feat2
    diff_vec_norm = norm(diff_vec)
    print 'diff_vec_norm: ', diff_vec_norm

    L2_sim = - np.dot(diff_vec, diff_vec)

    return L2_sim


def calc_similarity_norm_L2(feat1, feat2):
    feat1_norm = norm(feat1)
    feat2_norm = norm(feat2)
    print 'feat1_norm: ', feat1_norm
    print 'feat2_norm: ', feat2_norm

    norm_diff_vec = feat1 / feat1_norm - feat2 / feat2_norm
    norm_diff_vec_norm = norm(norm_diff_vec)
    print 'norm_diff_vec_norm: ', norm_diff_vec_norm

    L2_sim = - np.dot(norm_diff_vec, norm_diff_vec)

    return L2_sim


def main(argv):
    args = parse_arguments(argv)
    print '===> args:\n', args

    save_dir = args.save_dir
    if not osp.exists(save_dir):
        os.makedirs(args.save_dir)

    root_dir1 = args.root_dir1
    if args.feat_list1_file.endswith('.json'):
        root_dir1 = osp.dirname(args.feat_list1_file)

    root_dir2 = args.root_dir2
    if args.feat_list2_file.endswith('.json'):
        root_dir2 = osp.dirname(args.feat_list2_file)

    feat_fn_list1, face_img_fn_list1 = get_feat_fn_list(args.feat_list1_file)
    feat_fn_list2, face_img_fn_list2 = get_feat_fn_list(args.feat_list2_file)

    fn_rlt = osp.join(save_dir, 'compare_rlt.txt')
    fp_rlt = open(fn_rlt, 'w')
    fp_rlt.write("{}\n\n".format(args))

    for (i, feat_fn1) in enumerate(feat_fn_list1):
        if face_img_fn_list1:
            face_img_fn = face_img_fn_list1[i]
        else:
            face_img_fn = None

        print "---> %s:\n" % feat_fn1
        fp_rlt.write("---> %s:\n" % feat_fn1)

        feat1, face_img1 = load_feature(feat_fn1, face_img_fn, root_dir1)

        if feat1 is None:
            print "    Failed to load\n"
            fp_rlt.write("    Failed to load\n")
            continue

        for (j, feat_fn2) in enumerate(feat_fn_list2):
            if face_img_fn_list2:
                face_img_fn = face_img_fn_list2[j]
            else:
                face_img_fn = None

            print "---> %s:\n" % feat_fn2
            fp_rlt.write("---> %s:\n" % feat_fn2)

            feat2, face_img2 = load_feature(feat_fn2, face_img_fn, root_dir2)

            if feat2 is None:
                print "    Failed to load\n"
                fp_rlt.write("    Failed to load\n")
                continue

            sim = calc_similarity(feat1, feat2)
            #sim = calc_similarity_L2(feat1, feat2)
            print '===> similarity: ', sim
            fp_rlt.write("===> similarity: %5.4f\n\n" % sim)

            if face_img1 is not None and face_img2 is not None:
                img_pair = np.hstack((face_img1, face_img2))

                img_pair_fn = '%s_vs_%s_%5.4f.jpg' % (
                    osp.basename(feat_fn1), osp.basename(feat_fn2), sim)
                img_pair_fn = osp.join(save_dir, img_pair_fn)

                sim_txt = '%5.4f' % sim
                cv2_put_text_to_image(
                    img_pair, sim_txt, 40, 5, 30, (0, 0, 255))
                cv2.imwrite(img_pair_fn, img_pair)

    fp_rlt.close()


if __name__ == '__main__':
    argv = []
    if len(sys.argv) < 2:
        feat_list1 = './list_npy_renlianku_20_fixbug.txt'
        feat_list2 = './list_npy_qiniu_staff_fixbug.txt'

        argv.append(feat_list1)
        argv.append(feat_list2)

        save_dir = './2_feat_list_compare_rlt_fixbug_eltavg'
        argv.append('--save_dir=' + save_dir)

#        root_dir1=''
#        root_dir2=''
#        argv.append('--root_dir1='+root_dir1)
#        argv.append('--root_dir2='+root_dir2)
    else:
        argv = sys.argv[1:]

    main(argv)
