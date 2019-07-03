### This file fits a multinomial regression model without including the difference features.

## packages and functions
# import require packages
require(dplyr)
require(glmnet)
require(ggplot2)

# function flat.accuracy
# takes two vectors prediction and reference
# calculates the proportion of results in prediction that are equal to those in reference
flat.accuracy <- function(prediction, reference) {
  sum(prediction == reference) / length(reference)
}


## Prepare the modified training data
# load the training set
load("Objects/train.set")

# remove a few observations for which the descriptor calculations appear to have thrown errors
train.set <- train.set[!(train.set$song %in% c("O - Coldplay", "F.O.D. - Green Day", "Living the Laws - Choking Victim", "M1A1 (Lil' Dub Chefin') - Gorillaz", "Call That Gone? - Paul Westerberg", "VCR Wheels - Tyler_ The Creator")), ]

# generate a list of the parent names for the time-series features
features <- unique(sub(pattern = "\\..*", replacement = "", grep("\\..*", colnames(train.set), value = TRUE)))

# reorder the features such that *.1 through *.10 are in order
train.set <- train.set[, c("genre", "song", "album", "length", outer(1:10, features, function(x, y) {paste(y, x, sep = ".")}))]

# for each pair of time-series features n and n+1, compute |n+1 - n|
for (i in 1:length(features)) {
  for (j in 1:9)
    train.set[, paste(features[i], "dif", j, sep = ".")] <- abs(train.set[, paste(features[i], j+1, sep = ".")] - train.set[, paste(features[i], j, sep = ".")])
}

# log/root transform features with significant right skew
train.set <- mutate_at(train.set, vars(one_of(c("length", 
                                                outer(c("kurtosis", "Q25", "roughness", "skewness", "zcr"), 1:10, paste, sep = ".")))), 
                       function(x) {log(x + .0001)})
train.set <- mutate_at(train.set, vars(one_of(c("length", 
                                                outer(c("amplitude.dif", "centroid.dif", "entropy.dif", "kurtosis.dif", "Q25.dif", "Q75.dif", "skewness.dif", "zcr.dif"), 1:9, paste, sep = ".")))), 
                       function(x) {(x^(1/4))})
train.set <- mutate_at(train.set, vars(one_of(c("length", 
                                                outer("flatness.dif", 1:9, paste, sep = ".")))), 
                       function(x) {(x^(1/2))})
train.set <- mutate_at(train.set, vars(one_of(c("length", 
                                                outer("roughness.dif", 1:9, paste, sep = ".")))), 
                       function(x) {(x^(1/10))})

# power transform features with significant left skew
train.set <- mutate_at(train.set, vars(one_of(
  outer(c("entropy"), 1:10, paste, sep = "."))), 
  function(x) {x^(20)})
train.set <- mutate_at(train.set, vars(one_of(
  outer(c("flatness"), 1:10, paste, sep = "."))), 
  function(x) {x^(4)})
train.set <- mutate_at(train.set, vars(one_of(
  outer(c("Q75"), 1:10, paste, sep = "."))), 
  function(x) {x^(3)})

# scale all numeric features to N(0, 1) distribution
# rebuild dataframe with scaled columns
train.set <- cbind(select(train.set, song, album, genre), sapply(select(train.set, -song, -album, - genre), scale))



## fit the regression
# set seed for reproducible cv group splitting
set.seed(239913)

# divide training set into ten cross-validation groups
cv.groups <- caret::createFolds(train.set$genre, k = 10)

# select lambda values to use for the regularization penalty (same as in fully featured model)
lambda.values <- rev(c(seq(1, 10, 2) %o% exp(-20:5)))

# generate a list of the parent names for the time-series features
features <- unique(sub(pattern = "\\..*", replacement = "", grep("\\..*", colnames(train.set), value = TRUE)))

# declare a list to hold the training results
glm.training.results <- list()

# fit the multinomial regression with a truncated dataset
for (i in 1:length(cv.groups)) {
  # split out the current cv group from the rest of the training set
  # explicitly select the non-difference features
  train.set.test <- select(train.set, "genre", "album", "length", outer(features, 1:10, paste, sep = "."))[cv.groups[[i]], ]
  train.set.train <- select(train.set, "genre", "album", "length", outer(features, 1:10, paste, sep = "."))[-cv.groups[[i]], ]
  
  # fit a multinomial regression with elastic net penalization on the 9/10 set
  # using a single alpha value (0.3) from the full training set optimization
  intermediate.cv.model <- glmnet(x = as.matrix(select(train.set.train, -genre, -album)), 
                                  y = train.set.train$genre, family = "multinomial", 
                                  alpha = 0.3, lambda = lambda.values, standardize = FALSE)
  
  # predict with the fitted model on the 1/10 set
  glm.training.results[[i]] <- predict(intermediate.cv.model, as.matrix(select(train.set.test, -genre, -album)), type = "class")
}



## compile the cross-validation results

# format the cross-validation results into a single dataframe
# declare an empty array to hold the compiled predictions
glm.training.results.compiled <- array(dim = c(dim(train.set)[1], length(lambda.values)))

# convert the list of predictions seperated by cv group into a single array
for (i in 1:length(glm.training.results)) {
  for (j in 1:length(lambda.values)) {
    glm.training.results.compiled[cv.groups[[i]], j] <- glm.training.results[[i]][, j]
  }
}

# compute classification accuracy for each condition
glm.tuning.scores <- apply(glm.training.results.compiled, 2, FUN = function(x) {flat.accuracy(x, ref = train.set$genre)})

# bind the scores into a dataframe with the lambda and alpha values
glm.tuning.scores <- data.frame(cbind(rev(lambda.values), glm.tuning.scores))
colnames(glm.tuning.scores) <- c("lambda", "score")

# plot flat accuracy versus lambda parameter
ggplot(glm.tuning.scores) + geom_point(aes(x = log(lambda), y = score))

# find maximum classification accuracy 
max(glm.tuning.scores$score)
