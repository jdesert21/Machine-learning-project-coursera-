---
title: "module8project"
author: "Jean-Daniel"
date: "30 avril 2018"
output: html_document
---

##Overview
One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants to predict the manner in which they did the exercise.

##Data download
```{r download,cache=TRUE}
temp <- tempfile()
download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv",temp)
dat0<-read.csv(temp)
unlink(temp)
temp <- tempfile()
download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv",temp)
dat1<-read.csv(temp)
unlink(temp)
```

##Data preparing
The downloaded datasets have 160 variables. We'll remove all unnecessaries variables : 

 * variables with a lot of missing values
 * variables not related to movements : timestamp, new_window

We'll then prepare a training and testing set. 
The set used for the quiz is named simply quiz.
```{r cleaning, warning=FALSE, message=FALSE}
library(dplyr)
library(caret)
cleandat<-dplyr::select(dat0,-matches("timestamp|kurtosis|skewness|max_|min_|amplitude_|var_|avg_|stddev_"),-c(X,new_window))
quiz<-dplyr::select(dat1,-matches("timestamp|kurtosis|skewness|max_|min_|amplitude_|var_|avg_|stddev_"),-c(X,new_window))
set.seed(323)
inTrain<-createDataPartition(y = cleandat$classe,p=0.75,list=FALSE)
training<-cleandat[inTrain,]
testing<-cleandat[-inTrain,]
```
Now, the datasets we'll use contains only 55 variables.

##Model fitting
We're going to use bagging to build our first model. 
To do so, we'll use the random forest function with the mtry parameter equals to the number of predictors.
```{r modelfit, warning=FALSE, message=FALSE}
library(caret)
library(parallel)
library(doParallel)
library(randomForest)
cluster <- makeCluster(detectCores() - 1) # convention to leave 1 core for OS
registerDoParallel(cluster)
fitControl <- trainControl(method = "cv",number = 5,allowParallel = TRUE)
set.seed(324)
bag1<-randomForest(classe~.,data=training,mtry=54,importance=TRUE)
bag1
```
The number of variables tried at each split is equal to the number of predictors, so we were right to pick bagging. 
The OOB (out of the bag) estimate error rate is very good : 0.39%. 
We'll just try to see if it's a good estimate based on the testing dataset.
But before that, we'll see what are the most important variables :
```{r gini}
varImpPlot(bag1,type = 2)
```
It would be a good idea to try a new fit without the least important variables to see what the changes are.

```{r bagging1}
bag1pred<-predict(bag1,newdata=testing)
table(bag1pred,testing$classe)

```
(1394+944+851+797+893)/4904 = 99,5% -> Error rate on testing dataset = 0,5%. The previous estimate was very accurate.
```{r bagging2}
bag1quiz<-predict(bag1,newdata=quiz)
stopCluster(cluster)
registerDoSEQ()
bag1quiz
```
##Conclusion
Those predictions give me a 100% score on the project quiz. 
The model seems pretty stable according the accuracy on training, testing and the result on the quiz. 
