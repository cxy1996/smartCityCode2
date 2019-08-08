#! /usr/bin/bash

for i in 1
do
DATASET_PATH=./dataset/fishEye/scene${i}
WOKESAPCE_PATH=./workspace/fishEyeMask/scene${i}
SIFT_RATIO=0.8
MIN_INLIERS=15
NUM_THREADS=12
GPU_INDEX=-1

if [ ! -d $WOKESAPCE_PATH ]; then
    mkdir $WOKESAPCE_PATH
fi

colmap feature_extractor \
    --database_path $WOKESAPCE_PATH/database.db \
    --image_path $DATASET_PATH \
    --ImageReader.single_camera 1 \
    --ImageReader.camera_model OPENCV_FISHEYE \
    --SiftExtraction.num_threads $NUM_THREADS \
    --SiftExtraction.gpu_index $GPU_INDEX

colmap exhaustive_matcher \
    --database_path $WOKESAPCE_PATH/database.db \
    --SiftMatching.max_ratio $SIFT_RATIO \
    --SiftMatching.min_num_inliers $MIN_INLIERS \
    --SiftMatching.num_threads $NUM_THREADS \
    --SiftMatching.gpu_index $GPU_INDEX

if [ ! -d $WOKESAPCE_PATH/sparse ]; then
    mkdir $WOKESAPCE_PATH/sparse
fi

colmap mapper \
    --database_path $WOKESAPCE_PATH/database.db \
    --image_path $DATASET_PATH \
    --output_path $WOKESAPCE_PATH/sparse \
    --Mapper.num_threads $NUM_THREADS\
    --Mapper.ba_global_pba_gpu_index $GPU_INDEX\

colmap model_converter \
    --input_path $WOKESAPCE_PATH/sparse/0 \
    --output_path $WOKESAPCE_PATH/sparse/ \
    --output_type TXT
done

python3 ./scripts/colmap/cloundAlign.py --methodType fishEyeMask --scene 1
