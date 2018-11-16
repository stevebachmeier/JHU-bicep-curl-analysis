# ========================================
# LOAD LIBRARIES
# ========================================
library(caret)
library(parallel)
library(doParallel)
library(dplyr)
library(GGally)

# ========================================
# LOAD/CLEAN DATA
# ========================================
testRaw <- read.csv("Data/pml-testing.csv")
trainingRaw <- read.csv("Data/pml-training.csv")

# Compare test and training sets
# Number of training set variables not in test set
sum(!(names(trainingRaw) %in% names(testRaw))) 
names(trainingRaw)[!(names(trainingRaw) %in% names(testRaw))] # Unique training set variable
names(testRaw)[!(names(trainingRaw) %in% names(testRaw))] # Unique test set variable

# Find all not-all-NA columns in test set
keepVars <- names(Filter(f=function(x) {!all(is.na(x))}, testRaw))

# Remove irrelevant columns
keepVars <- keepVars[-c(1, 2, 3, 4, 5, 6, 7, 60)]

# Create working df
working <- select(trainingRaw, keepVars, classe)

# Check for NAs in working set
sum(is.na(working)) # None

# Convert integers to numerics
working <- mutate_all(working, function(x) {if(is.integer(x)) as.numeric(x) else x})
test <- mutate_all(testRaw, function(x) {if(is.integer(x)) as.numeric(x) else x})

# Check for NAs in test set keepVars
sum(is.na(select(test, keepVars)))

# ========================================
# VARIABLE REDUCTION
# ========================================
# Check for near-zero variance variables
sum(nearZeroVar(working, saveMetrics=TRUE)[,4])

# Check for high correlation variables to classe
workingNum <- working
workingNum$classe <- as.numeric(workingNum$classe)
max(abs(cor(workingNum)[1:ncol(workingNum) - 1,ncol(workingNum)])) # < 0.5
plot(abs(cor(workingNum)[1:(ncol(workingNum) - 1),ncol(workingNum)]), ylim=c(0,1),
     main="Variable-class correlation", ylab="|Cor|") 
abline(h=0.5, col="red")

# Check for high correlation between variables
varMat <- workingNum[, 1:(ncol(workingNum) - 1)]
corVarMat <- abs(cor(varMat))
corVarMat[!lower.tri(corVarMat)] <- 0
max(corVarMat) # > 0.5
plot(corVarMat[lower.tri(corVarMat)], ylim=c(0,1),
     main="Variable-variable correlation", ylab="|Cor|")
abline(h=0.5, col="red")

# Find variables with low inter-varable correlation
corReqt <- 0.5
lowCorVars <- names(data.frame(corVarMat[, apply(corVarMat, 2, function(x) {all(x < corReqt)})]))

# Subset working data set to only include lowCorVars
working <- select(working, lowCorVars, classe)
corVarWorkingMat <- abs(cor(working[, 1:(ncol(working) - 1)]))
corVarWorking <- corVarWorkingMat[lower.tri(corVarWorkingMat)]
plot(corVarWorking, ylim=c(0,1), main="Working set variable-variable correlation",
     ylab="|Cor|")
abline(h=0.5, col="red")

# ========================================
# EXPLORATORY DATA ANALYSIS
# ========================================
working2 <- working
working2$classe <- as.numeric(working2$classe)
ggpairs(data=working2, lower=list(continuous="smooth"))

# ========================================
# TRAIN/TEST SPLIT
# ========================================
set.seed(101)
inTrain <- createDataPartition(y=working$classe, p=0.75, list=F)
training <- working[inTrain,]
testing <- working[-inTrain,]

# ========================================
# ANALYSIS
# ========================================

# ---- MODEL FITTING ----

# ---- rpart (decision tree) ----
set.seed(101)
fitRpart <- train(classe ~ ., data=training, method="rpart")
predRpart <- predict(fitRpart, newdata=testing)

# ---- rf (random forest) ----
# fitRpart <- train(classe ~ ., data=training, method="rf", prox=T)

# 2 fold
number <- 2
cluster <- makeCluster(detectCores() - 1)
registerDoParallel(cluster)
set.seed(101)
fitRf2 <- train(classe ~ ., data=training, method="rf", 
    trControl=trainControl(method="cv", number=number, allowParallel=T))
stopCluster(cluster)
registerDoSEQ()
predRf2 <- predict(fitRf2, newdata=testing)

# 3 fold
number <- 3
cluster <- makeCluster(detectCores() - 1)
registerDoParallel(cluster)
set.seed(101)
fitRf3 <- train(classe ~ ., data=training, method="rf", 
                trControl=trainControl(method="cv", number=number, allowParallel=T))
stopCluster(cluster)
registerDoSEQ()
predRf3 <- predict(fitRf3, newdata=testing)

# 5 fold
number <- 5
cluster <- makeCluster(detectCores() - 1)
registerDoParallel(cluster)
set.seed(101)
fitRf5 <- train(classe ~ ., data=training, method="rf", 
                trControl=trainControl(method="cv", number=number, allowParallel=T))
stopCluster(cluster)
registerDoSEQ()
predRf5 <- predict(fitRf5, newdata=testing)

# ---- gbm (boosting) ----

# 2 fold
number <- 2
cluster <- makeCluster(detectCores() - 1)
registerDoParallel(cluster)
set.seed(101)
fitGbm2 <- train(classe ~ ., data=training, method="gbm", verbose=F, 
    trControl=trainControl(method="cv", number=number, allowParallel=T))
stopCluster(cluster)
registerDoSEQ()
predGbm2 <- predict(fitGbm2, newdata=testing)

# 3 fold
number <- 3
cluster <- makeCluster(detectCores() - 1)
registerDoParallel(cluster)
set.seed(101)
fitGbm3 <- train(classe ~ ., data=training, method="gbm", verbose=F, 
                 trControl=trainControl(method="cv", number=number, allowParallel=T))
stopCluster(cluster)
registerDoSEQ()
predGbm3 <- predict(fitGbm3, newdata=testing)

# ---- lda ----
set.seed(101)
fitLda <- train(classe ~ ., data=training, method="lda")
predLda <- predict(fitLda, testing)

# ---- bn (Naive-Bayes) ----
cluster <- makeCluster(detectCores() - 1)
registerDoParallel(cluster)
set.seed(101)
fitNb <- train(classe ~ ., data=training, method="nb")
stopCluster(cluster)
registerDoSEQ()
predNb <- predict(fitNb, testing)

# ---- Accuracies on test set ----
accDF <- data.frame(
    tree=confusionMatrix(predRpart, testing$classe)$overall["Accuracy"],
    rf2=confusionMatrix(predRf2, testing$classe)$overall["Accuracy"],
    rf3=confusionMatrix(predRf3, testing$classe)$overall["Accuracy"],
    rf5=confusionMatrix(predRf5, testing$classe)$overall["Accuracy"],
    gbm2=confusionMatrix(predGbm2, testing$classe)$overall["Accuracy"],
    gbm3=confusionMatrix(predGbm3, testing$classe)$overall["Accuracy"],
    lda=confusionMatrix(predLda, testing$classe)$overall["Accuracy"],
    nb=confusionMatrix(predNb, testing$classe)$overall["Accuracy"])

accDF

# ---- pca ----

# random forest, 2 fold
number <- 2
set.seed(101)
pca <- preProcess(select(training, -classe), method="pca")
trainPca <- predict(pca, select(training, -classe))
cluster <- makeCluster(detectCores() - 1)
registerDoParallel(cluster)
set.seed(101)
fitPcaRf2 <- train(x=trainPca, y=training$classe, method="rf", 
    trControl=trainControl(method="cv", number=number, allowParallel=T))
stopCluster(cluster)
registerDoSEQ()
testPcaRf <- predict(pca, select(testing, -classe))
predPcaRf2 <- predict(fitPcaRf2, testPcaRf)

# random forest, 3 fold
number <- 3
set.seed(101)
pca <- preProcess(select(training, -classe), method="pca")
trainPca <- predict(pca, select(training, -classe))
cluster <- makeCluster(detectCores() - 1)
registerDoParallel(cluster)
set.seed(101)
fitPcaRf3 <- train(x=trainPca, y=training$classe, method="rf", 
                   trControl=trainControl(method="cv", number=number, allowParallel=T))
stopCluster(cluster)
registerDoSEQ()
testPcaRf <- predict(pca, select(testing, -classe))
predPcaRf3 <- predict(fitPcaRf3, testPcaRf)

# ---- Update accuracy table ----
accDF$pcaRf2 <- confusionMatrix(predPcaRf2, testing$classe)$overall["Accuracy"]
accDF$pcaRf3 <- confusionMatrix(predPcaRf3, testing$classe)$overall["Accuracy"]
accDF

# ---- ENSEMBLE ----

# Train/test/validation split
set.seed(101)
inValidation <- createDataPartition(y=working$classe, p=0.20, list=F)
validation <- working[inValidation,]
workingV <- working[-inValidation,]
set.seed(101)
inTrainV <- createDataPartition(y=workingV$classe, p=0.75, list=F)
trainingV <- workingV[inTrainV,]
testingV <- workingV[-inTrainV,]

# rf2 + gbm2
# rf2
number <- 2
cluster <- makeCluster(detectCores() - 1)
registerDoParallel(cluster)
set.seed(101)
fitRf2_2 <- train(classe ~ ., data=trainingV, method="rf", 
                trControl=trainControl(method="cv", number=number, allowParallel=T))
stopCluster(cluster)
registerDoSEQ()
predRf2_2 <- predict(fitRf2_2, newdata=testingV)
predRf2_2V <- predict(fitRf2_2, newdata=validation)

# gbm2
number <- 2
cluster <- makeCluster(detectCores() - 1)
registerDoParallel(cluster)
set.seed(101)
fitGbm2_2 <- train(classe ~ ., data=trainingV, method="gbm", verbose=F, 
                 trControl=trainControl(method="cv", number=number, allowParallel=T))
stopCluster(cluster)
registerDoSEQ()
predGbm2_2 <- predict(fitGbm2_2, newdata=testingV)
predGbm2_2V <- predict(fitGbm2_2, newdata=validation)

# Stack
predDFRf2Gbm2 <- data.frame(predRf2_2, predGbm2_2, classe=testingV$classe)
fitStackRf2Gbm2 <- train(classe ~ ., data=predDFRf2Gbm2, method="gam")
predDFRf2Gbm2V <- data.frame(predRf2_2=predRf2_2V, predGbm2_2=predGbm2_2V)
predRf2Gbm2V <- predict(fitStackRf2Gbm2, predDFRf2Gbm2V)

# ---- Update accuracy table ----
accDF$stackRf2Gbm2 <- confusionMatrix(predRf2Gbm2V, validation$classe)$overall["Accuracy"]
rbind(round(accDF, 2), "OOS Error"=round(sapply(accDF, function(x) {1-x}), 2))

# ---- PREDICT ON TEST SET ----
# 2 trees
predRf2_test <- predict(fitRf2, newdata=test)