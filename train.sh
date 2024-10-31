#!/bin/bash
python run.py \
    --frame_dir /Users/jialinli//Desktop/GraduateSchool/Fall2024/EN705_643_DeepLearning_Pytorch/Video_Classification/datasets/preprocessed_data/HMDB51 \
    --train_size 0.75 \
    --test_size 0.15 \
    --model_type lrcn \
    --n_classes 51 \
    --fr_per_vid 16 \
    --batch_size 4 \
    --mode train
