@echo off
echo "started"
net_run evaluation -c C:\Users\CCIG\Anaconda3\envs\SEGM\Scripts\tests\context_patch_size_tests\cv\context_unet_cv1_eval.ini -a niftynet.application.segmentation_application.SegmentationApplication
net_run evaluation -c C:\Users\CCIG\Anaconda3\envs\SEGM\Scripts\tests\context_patch_size_tests\cv\context_unet_cv2_eval.ini -a niftynet.application.segmentation_application.SegmentationApplication
net_run evaluation -c C:\Users\CCIG\Anaconda3\envs\SEGM\Scripts\tests\context_patch_size_tests\cv\context_unet_cv3_eval.ini -a niftynet.application.segmentation_application.SegmentationApplication
net_run evaluation -c C:\Users\CCIG\Anaconda3\envs\SEGM\Scripts\tests\context_patch_size_tests\cv\context_unet_cv4_eval.ini -a niftynet.application.segmentation_application.SegmentationApplication
net_run evaluation -c C:\Users\CCIG\Anaconda3\envs\SEGM\Scripts\tests\context_patch_size_tests\cv\context_unet_cv5_eval.ini -a niftynet.application.segmentation_application.SegmentationApplication
echo 