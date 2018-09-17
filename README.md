# Caffe Feature Extractor
A wrapper for extractring features from Caffe network, with a config file to define network parameter.

## Contents
```
---
|- caffe_feature_extractor  # caffe feature extractor
|- scripts  # test scripts
    |- extract_features_for_image_list 
    |- face_feature_extractor   # detect faces using MTCNN, align faces and extract features
|- utils                    # utils for compare similarity between two features
---
```

## Python requirements:
```
pycaffe
numpy
skimage
json
```