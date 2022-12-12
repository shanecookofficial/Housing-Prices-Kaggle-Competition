# Housing Prices Kaggle Competition

## Overview

### Software Description

This Github repo holds my progress on improving my Machine Learning model for the "Housing Prices Competition for Kaggle Learn Users". You can find this competition and make your own submission at [Kaggle.com](https://www.kaggle.com/competitions/home-data-for-ml-course).

All data is not mine and is owned and distributed by Kaggle and can be found in the data folder of this repo or on the site refferenced above.

As of right now, I am currently working with the Pandas and Sklearn libraries to form my predictions model. The whole idea behind it was to develop a predictions model capable of predicting the price of a home based off of certain data characteristics such as the number of bedrooms, square footage, etc.

The first main chunk is pulling the data from a csv file and storing it into a Pandas Data Frame. This makes it really easy to work with because I can use the .describe, .head, and .columns functions to visualize the data.

The second main chunk splits the data up into testing and validation data so I can train my model to eventually be used with a similar data set. The train_test_split function from Sklearn is able to perform that split. I then can create a Random Forrest which essentially is a bunch of decision trees to fit the test data to it.

The last chunk then takes the fitted data and predicts agains the validation data of the price, and uses the mean_absolute_error function from Sklearn to determing what the mean absolute error is.

Please see comments for more detail.

### Software Purpose
Machine learning is always a topic that has interested me and I always felt it was a tough field to break into. I wanted to write my first model to get a feel for the kind of work that gets done. The purpose of this software and the repository in general is to be an evolving repo for this competition to benchmark my progress in Data Science and Machine Learning. Overtime as I make better models that make better predictions, I will post updates here in the form of commits.

[Software Demo Video](https://youtu.be/Idk1QKfJT-k)

## Development Environment

### Tools used for development
* IDE: Visual Studio Code
* VCS: Git
* Repo Host: Github
* Data Set: Ames, Iowa home data set *provided by Kaggle*
* Testing: Jupyter Notebooks *provided by Kaggle*

I only used the Python programming language for this assignment. I used the pandas library to create my data frame for the main data. I used the sklearn library to create my random forest regressor model, as well as being able to split the data into training and validation data, and calculating the mean absolute error value of the data.

## Useful Websites

{Make a list of websites that you found helpful in this project}
* [Competition Page](https://www.kaggle.com/competitions/home-data-for-ml-course)
* [Intro to Machine Learning Course](https://www.kaggle.com/learn/intro-to-machine-learning)

## Future Work

* Work on improving the MAE to get a higher score, maybe use a different model than a decision tree or a forest?