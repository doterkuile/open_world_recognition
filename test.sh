#!/bin/bash

python setup_train_data.py config/TinyImageNet_train.yaml
python setup_test_data.py config/TinyImageNet_train.yaml
python setup_train_data_2.py config/TinyImageNet_train.yaml
python setup_test_data_2.py config/TinyImageNet_train.yaml

