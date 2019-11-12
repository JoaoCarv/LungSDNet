@echo off
echo "started"
REM -----cv1
net_run train -c C:\Users\CCIG\Anaconda3\envs\SEGM\Scripts\tests\context_patch_size_tests\cv\context_unet_cv1.ini -a niftynet.application.segmentation_application.SegmentationApplication
net_run inference -c C:\Users\CCIG\Anaconda3\envs\SEGM\Scripts\tests\context_patch_size_tests\cv\context_unet_cv1.ini -a niftynet.application.segmentation_application.SegmentationApplication
REM -----cv2
net_run train -c C:\Users\CCIG\Anaconda3\envs\SEGM\Scripts\tests\context_patch_size_tests\cv\context_unet_cv2.ini -a niftynet.application.segmentation_application.SegmentationApplication
net_run inference -c C:\Users\CCIG\Anaconda3\envs\SEGM\Scripts\tests\context_patch_size_tests\cv\context_unet_cv2.ini -a niftynet.application.segmentation_application.SegmentationApplication
REM -----cv3
net_run train -c C:\Users\CCIG\Anaconda3\envs\SEGM\Scripts\tests\context_patch_size_tests\cv\context_unet_cv3.ini -a niftynet.application.segmentation_application.SegmentationApplication
net_run inference -c C:\Users\CCIG\Anaconda3\envs\SEGM\Scripts\tests\context_patch_size_tests\cv\context_unet_cv3.ini -a niftynet.application.segmentation_application.SegmentationApplication
REM -----cv4
net_run train -c C:\Users\CCIG\Anaconda3\envs\SEGM\Scripts\tests\context_patch_size_tests\cv\context_unet_cv4.ini -a niftynet.application.segmentation_application.SegmentationApplication
net_run inference -c C:\Users\CCIG\Anaconda3\envs\SEGM\Scripts\tests\context_patch_size_tests\cv\context_unet_cv4.ini -a niftynet.application.segmentation_application.SegmentationApplication
REM -----cv5
net_run train -c C:\Users\CCIG\Anaconda3\envs\SEGM\Scripts\tests\context_patch_size_tests\cv\context_unet_cv5.ini -a niftynet.application.segmentation_application.SegmentationApplication
net_run inference -c C:\Users\CCIG\Anaconda3\envs\SEGM\Scripts\tests\context_patch_size_tests\cv\context_unet_cv5.ini -a niftynet.application.segmentation_application.SegmentationApplication

echo 
net_run evaluation -c C:\Users\CCIG\Anaconda3\envs\SEGM\Scripts\tests\context_patch_size_tests\cv\context_unet_cv1_eval.ini -a niftynet.application.segmentation_application.SegmentationApplication
net_run evaluation -c C:\Users\CCIG\Anaconda3\envs\SEGM\Scripts\tests\context_patch_size_tests\cv\context_unet_cv2_eval.ini -a niftynet.application.segmentation_application.SegmentationApplication
net_run evaluation -c C:\Users\CCIG\Anaconda3\envs\SEGM\Scripts\tests\context_patch_size_tests\cv\context_unet_cv3_eval.ini -a niftynet.application.segmentation_application.SegmentationApplication
net_run evaluation -c C:\Users\CCIG\Anaconda3\envs\SEGM\Scripts\tests\context_patch_size_tests\cv\context_unet_cv4_eval.ini -a niftynet.application.segmentation_application.SegmentationApplication
net_run evaluation -c C:\Users\CCIG\Anaconda3\envs\SEGM\Scripts\tests\context_patch_size_tests\cv\context_unet_cv5_eval.ini -a niftynet.application.segmentation_application.SegmentationApplication