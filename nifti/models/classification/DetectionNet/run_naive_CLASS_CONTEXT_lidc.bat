@echo off
echo "started"
net_run train -c naive_lidc_classification_net.ini -a niftynet.application.classification_nodules_application.RegressionApplication
REM net_run inference -c naive_lidc_classification_net_inf.ini -a niftynet.application.classification_nodules_application.RegressionApplication
python C:\.......\evaluate_classes.py
REM echo 

