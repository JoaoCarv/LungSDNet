@echo off
REM runs all the scripts in correct order to have dataset with smaller lesions.
echo "started resampling"
python select_small.py
python spacing_sizes_statistics_small.py
python prep_dataset_DECA_small.py
python addlayers_small.py
echo 
