# Medical Insurance Cost Predictor

## About

Project aimed to establish ETL based ML pipeline for prediction of medical insurance cost for US citizens.

## Contents
- **/src** - Contains analysis files
- **/models** - Contains trined model binary files
- **/data** - Contains data .csv files
- **/pipeline** - Contains production scripts

## Usage
**!ALL SCRIPTS ARE TO BE EXECUTED FROM PROJECT ROOT!**

Typical usage scenario consists of:
- provide train.csv into _/data_ folder and execute _/pipeline/train.py_ to train model
- provide new.csv without charges column into /data folder and execute _/pipeline/predict.py_ to predict charges
- outputs shall be written into _/data/results.csv_