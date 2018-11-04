rm(list=ls())
library(readr)
library(caret)

#1. Load the dataset insurance.csv into memory.
insurance <- read.csv("C:\\Users\\nidhi\\Downloads\\insurance.csv")
str(insurance)

#2. Covert the following predictors to factor using function factor()
#a) sex b) smoker c) region
insurance$sex <-factor(insurance$sex)
insurance$smoker <-factor(insurance$smoker)
insurance$region <-factor(insurance$region)

str(insurance)

#3. Perform a 80%/20% train/test split of the dataset.

set.seed(2018)
inTrain <- createDataPartition(
  y = insurance$charges,
  p = .8,
  list = FALSE
)
train.data <- insurance[ inTrain,]
test.data  <- insurance[-inTrain,]
nrow(train.data) # 1072
nrow(test.data) # 266

#4. Linear models with regularization :Ridge regression
#a. Perform ridge regression with lambda= c(0.001,0.01, 0.1,1,10,100,1000,10000) using training data set with 5 fold CV.
library(glmnet)
lambda <- c(0.001,0.01, 0.1,1,10,100,1000,10000)
set.seed(2018)

ridge <- train(
  charges~.,data=train.data,method="glmnet",
  trControl=trainControl("cv",number=5) # total folds=5 in cross validation
  ,preProcess="scale"
  ,tuneGrid=expand.grid(alpha=0,lambda=lambda) #Alpha=0 then Ridge
)
#b. What's the lambda for the best model?

ridge$bestTune$lambda
#100
coef(ridge$finalModel,ridge$bestTune$lambda)
#c. Compute the RMSE and R squared for the test dataset.

predictions <-predict(ridge,test.data)

data.frame(
  RMSE=RMSE(predictions,test.data$charges),
  RSQUARE=R2(predictions,test.data$charges)
)
# RMSE      RSQUARE
# 6467.231  0.7599585



#5. Linear models with regularization :Lasso regression
#a. Perform lasso regression with lambda= c(0.001,0.01, 0.1,1,10,100,1000,10000) using training data set with 5 fold CV.

lambda <- c(0.001,0.01, 0.1,1,10,100,1000,10000)
set.seed(2018)
lasso <- train(
  charges~.,data=train.data,method="glmnet",
  trControl=trainControl("cv",number=5) # total folds=5 in cross validation
  ,preProcess="scale"
  ,tuneGrid=expand.grid(alpha= 1,lambda = lambda) #Alpha=1, then lasso
)

#b. What's the lambda for the best model?
lasso$bestTune$lambda
#10
coef(lasso$finalModel,lasso$bestTune$lambda)

#c. Compute the RMSE and R squared for the test dataset.

predictions <-predict(lasso,test.data)

data.frame(
  RMSE=RMSE(predictions,test.data$charges),
  RSQUARE=R2(predictions,test.data$charges)
)
# RMSE     RSQUARE
# 6324.589 0.7614842


#6. Linear models with regularization :Elastic Net regression

#a. Perform lasso regression with lambda= c(0.001,0.01, 0.1,1,10,100,1000,10000) And alpha =seq(0,1, length=20) using training data set with 5 fold CV.
alpha <- seq(0,1, length=20)
lambda <- c(0.001,0.01, 0.1,1,10,100,1000,10000)
set.seed(2018)
elastic <- train(
  charges~.,data=train.data,method="glmnet",
  trControl=trainControl("cv",number=5) # total folds=5 in cross validation
  ,preProcess="scale"
  ,tuneGrid=expand.grid(alpha=alpha,lambda=lambda)  
)

#b. What's the lambda for the best model?
elastic$bestTune$lambda #10

coef(elastic$finalModel,elastic$bestTune$lambda)
#c. Compute the RMSE and R squared for the test dataset.
predictions <-predict(elastic,test.data)

data.frame(
  RMSE=RMSE(predictions,test.data$charges),
  RSQUARE=R2(predictions,test.data$charges)
)
# RMSE   RSQUARE
# 6332.258 0.7611586
