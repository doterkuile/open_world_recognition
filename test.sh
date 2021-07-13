#!/bin/bash


python setup_test_data.py TinyImageNet_train.yaml
python setup_train_data_2.py TinyImageNet_train.yaml
python setup_test_data_2.py TinyImageNet_train.yaml

