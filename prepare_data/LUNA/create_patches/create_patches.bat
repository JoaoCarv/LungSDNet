@echo off
echo "started resampling"
python rs_nodule_size_statistic.py
python prep_dataset_LUNA.py
echo 
