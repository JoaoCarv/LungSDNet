@echo off
echo "started resampling"
REM 1. change path of dir + create resample dir + create images/masks dir
REM 2. need to change path in resample_size.py + create folders
REM 3 change path from cretae pathces
python ..\stats\size_spacing_statistics.py
python resample_size.py
python resample_size_segs.py
echo 
