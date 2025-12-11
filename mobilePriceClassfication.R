## Mobile Pricing Range Classification,
#can we predict mobile price range using hardware specifications provided by the data set?
#which models perform the best?
#Using GLM, Deep Learning (via Keras), Random Forest (w decision tree), SVM

##loading libraries needed
library(tidyverse)
library(ggplot2) #for plotting graphs
library(DataExplorer)
library(caret) 
library(nnet)
library(keras) #deep learning
library(randomForest)#random forest section
library(rpart)
library(rpart.plot)
print("loaded")

#loading csv file into a dataframe 
df <- read.csv("train.csv")

####################
##data exploration##
####################
str(df)
summary(df)

table <- table(df$price_range)# seeing the distributions
table
prop.table(table)

#checking for missing values
colSums(is.na(df)) #no missing values

#boxplot to compare values as price_range increase
#also to use to see which predictors have more of a contribution
plot_boxplot(df, by = "price_range")

##preparing the data
#this will be our target variable, price_range
df$price_range <- as.factor(df$price_range)

set.seed(2160)#setting a seed so we can reproduce the results

#setting up the train/test splits, using 70/30 here
trainIndex <- createDataPartition(df$price_range, p = 0.7, list = FALSE)

df_train <- df[train_index,] #70% training data
df_test <- df[-train_index, ] #30% test data
print("ready")

#############
##GLM Model##
#############

#10 fold cross validation
train_control <- trainControl(
  method ="cv",
  number = 10
)

#using a multinomial model as the outcome is categorical
glm_model <- train(
  price_range ~.,
  data = df_train,
  method = "multinom", 
  trControl = train_control
)

#lets us see the coefficients on the model
glm_model
summary(glm_model$finalModel)

#predicting on the test set
glm_pred <- predict(glm_model, newdata = df_test)

confusionMatrix(glm_pred, df_test$price_range)# confusion matrix

##############################
## Deep learning using Keras##
##############################

#reusing train/test split from GLM
trainIndex <- createDataPartition(df$price_range, p = 0.7, list = FALSE)

df_train <- df[train_index,] #training set
df_test <- df[-train_index, ] #test set

#preprocessing to scale the inputs
set.seed(2160)
preproc <- preProcess(df_train[, -which(names(df_train)=="price_range")],
                      method = c("center","scale"))

#applying the preprocessed values to the training and test sets
train_x <- predict(preproc, df_train[,-which(names(train)=="price_range")])
test_x <- predict(preproc, df_test[,-which(names(train)=="price_range")])

#getting the price range variables from each set 
train_y <- df_train$price_range
test_y <- df_test$price_range

#ensures values are categorical for keras 
#we need one hot encoded labels for classification
train_y_cat <- to_categorical(train_y, num_classes = 4)
test_y_cat <- to_categorical(test_y, num_classes = 4)

##building the deep learning model with keras
model <- keras_model_sequential() %>%
  #first layer, using 40 neurons
  #using relu activation to turn any negatives into a zero
  layer_dense(units = 40, activation = "relu", input_shape = ncol(train_x)) %>%
  layer_dropout(rate = 0.3) %>% #randomly removing 50% of the neurons to reduce overit
  #second layer: using half the neurons
  layer_dense(units = 20, activation = "relu") %>%
  layer_dropout(rate = 0.3) %>%
  #output, using 4 units to represent the classes
  layer_dense(units = 4, activation = "softmax")

model %>% compile(
  loss = "categorical_crossentropy", #classification so we use cross-entropy
  optimizer = optimizer_adam(),
  metrics = "accuracy" #tracking the accuracy during training
)

#training the model
history <- model %>% fit(
  as.matrix(train_x), #features used to train
  train_y_cat,#one hotting the labels, 0 or 1
  epochs = 60, #number of times we pass through the data
  batch_size = 32, #samples size per update
  validation_split = 0.1 # using 10% of the train set as validation set
)

plot(history) #visual representation

model %>% evaluate(as.matrix(test_x), test_y_cat)

#generating the predictions
pred_prob <- model %>% predict(as.matrix(test_x)) #
pred_class <- max.col(pred_prob) - 1 #converts labels to correspond to the class


#confusion matrix for analysis
confusionMatrix(as.factor(pred_class), as.factor(test_y))
####################
##random forest RF##
####################

#still using the same 70/30 train/test split from the other models
set.seed(2160)#sticking to the same seed for consistency

#training the random forest model
rf_model <- train(
  price_range ~ .,
  data = df_train,
  method = "rf", #random forest method
  trControl = trainControl(method = "cv", number = 10), #10 fold cv
  tuneLength = 10,
)

rf_tree1 <- randomForest::getTree(rf_model$finalModel, k = 1, labelVar = TRUE)

rf_tree_plot <- rpart(price_range ~., data = df_train, method ="class")

rpart.plot(rf_tree_plot,
           type = 2,
           extra = 104,
           fallen.leaves = TRUE,
           )
#predicting values for the confusion matrix using RF
rf_pred <- predict(rf_tree_plot, df_test, type = "class")
confusionMatrix(as.factor(rf_pred), as.factor(df_test$price_range))

#Variable Importance Plot 
rf_imp <- varImp(rf_model)
plot(rf_imp, top = 20)






