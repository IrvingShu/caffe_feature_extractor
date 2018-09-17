#/usr/bin/env python
# # -*- coding: utf-8 -*-
"""
@author: zhaoy
"""
import os
import sys
import os.path as osp

from compare_feats import compare_feats


def get_feat_fns(feat_dir, suffixes=None):
    if suffixes is None:
        suffixes = ['.bin', '.npy']
    fn_list = os.listdir(feat_dir)
    feat_fn_list = []

    for it in fn_list:
        ext = osp.splitext(it)[1]
        if ext in suffixes:
            feat_fn_list.append(osp.join(feat_dir, it))

    return feat_fn_list


def compare_feats_under_dir(feat_dir):
    feat_fn_list = get_feat_fns(feat_dir)

    num_fns = len(feat_fn_list)
    for i in range(num_fns):
        for j in range(i + 1, num_fns):
            compare_feats(feat_fn_list[i], feat_fn_list[j])


if __name__ == '__main__':
#    feat_dir = './test_feats'
    feat_dir = './rlt_feat_face_chips/fc5'

    if len(sys.argv) > 1:
        feat_dir = sys.argv[1]

    compare_feats_under_dir(feat_dir)
