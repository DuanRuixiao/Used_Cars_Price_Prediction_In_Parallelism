# Used Cars Price Prediction In Parallelism

## Report

Completed report can be accessed under "reports" directory.

## Introduction

This project trains a model to predict used cars price given some features such as name, location, year, driven distance, fuel type, transmission, engine, power, seats and new car's price.


* Took overview and analyzed row dataset.
* Preprocessed row dataset including text preprocessing, binarization and normalization in parallel.
* Trained linear regression model in `pyspark.ml` package.
* Tuned hyperparameters to improve the performance of model.
* Evaluated model by k folds cross-validation.
* Supposed some advice to boost model.
