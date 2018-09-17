#!/bin/bash

export PYTHONPATH=/usr/local/lib/python2.7/dist-packages:$PYTHON
export MXNET_CUDNN_AUTOTUNE_DEFAULT=0

#change your model name
#feature save
#not modified
SAVE_ROOT="/workspace/data/face-idcard-1M/features/"

#change name
SAVE_NAME="model-r18-spa-m2.0-8gpu-v15-and-msra-154666-ep233-caffe"
SAVE_DIR="${SAVE_ROOT}${SAVE_NAME}"

#log
LOG_NAME="./logs/model-r18-spa-m2.0-8gpu-v15-and-msra-ep233-caffe-"

GPU_NUM=8

for ((i=0; i<$GPU_NUM; i++))
    do
        /bin/sleep 2
        nohup python ./extract_features_for_idcard.py \
              --config-json=./extractor_config_resnet18_pod.json \
              --image-list="/workspace/data/qyc/qyc_work/ipca_qyc/data/multi_scene/"${i}".txt" \
              --image-dir=/workspace/data/face-idcard-1M/face-idcard-1M-mtcnn-aligned-112x112/aligned_imgs \
              --save-dir=$SAVE_DIR \
              --gpu=$i \
               > ${LOG_NAME}"split"${i}".txt" 2>&1 &
    done
