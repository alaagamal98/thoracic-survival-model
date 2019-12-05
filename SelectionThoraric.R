library(caTools) #splitting dataset  
library(e1071) #Naive bayes
library(ElemStatLearn) #element statical learning
library(class) #classification
library(caret) #cross validation
library(glmnet) #logistic regression
library(plyr)
library(dplyr) 
library(ipred)
library(skimr)
library(foreach)
library(doParallel)
library(dummies)


# Importing the dataset0
dataset <- read.csv(file="ThoraricSurgery.csv")
dataset1 <- read.csv(file="ThoraricSurgery.csv")


# Encoding the features as factors for dataset
dataset$Risk1Yr = factor(dataset$Risk1Yr,levels = c('TRUE','FALSE'),labels = c(1,0))
dataset$PainBS = factor(dataset$PainBS,levels = c('TRUE','FALSE'),labels = c(1,0))
dataset$HaemoptysisBS = factor(dataset$HaemoptysisBS,levels = c('TRUE','FALSE'),labels = c(1,0))
dataset$DyspnoeaBS = factor(dataset$DyspnoeaBS,levels = c('TRUE','FALSE'),labels = c(1,0))
dataset$CoughBS = factor(dataset$CoughBS,levels = c('TRUE','FALSE'),labels = c(1,0))
dataset$WeaknessBS = factor(dataset$WeaknessBS,levels = c('TRUE','FALSE'),labels = c(1,0))
dataset$Type2Diabetes = factor(dataset$Type2Diabetes,levels = c('TRUE','FALSE'),labels = c(1,0))
dataset$HeartAttack6M = factor(dataset$HeartAttack6M,levels = c('TRUE','FALSE'),labels = c(1,0))
dataset$PeripheralArterialDiseases = factor(dataset$PeripheralArterialDiseases,levels = c('TRUE','FALSE'),labels = c(1,0))
dataset$Smoking = factor(dataset$Smoking,levels = c('TRUE','FALSE'),labels = c(1,0))
dataset$Asthma = factor(dataset$Asthma,levels = c('TRUE','FALSE'),labels = c(1,0))
dataset$DGN = factor(dataset$DGN,levels = c('DGN1','DGN2','DGN3','DGN4','DGN5','DGN6','DGN8'),labels = c(1,2,3,4,5,6,8))
dataset$PerformanceStatus = factor(dataset$PerformanceStatus,levels = c('PRZ0','PRZ1','PRZ2'),labels = c(0,1,2))
dataset$SizeOfTumer = factor(dataset$SizeOfTumer,levels = c('OC11','OC12','OC13','OC14'),labels = c(11,12,13,14))

#Converting Factors to Numeric for dataset
dataset$Risk1Yr = as.numeric(levels(dataset$Risk1Yr))[dataset$Risk1Yr]
dataset$PainBS = as.numeric(levels(dataset$PainBS))[dataset$PainBS]
dataset$HaemoptysisBS = as.numeric(levels(dataset$HaemoptysisBS))[dataset$HaemoptysisBS]
dataset$DyspnoeaBS = as.numeric(levels(dataset$DyspnoeaBS))[dataset$DyspnoeaBS]
dataset$CoughBS = as.numeric(levels(dataset$CoughBS))[dataset$CoughBS]
dataset$WeaknessBS = as.numeric(levels(dataset$WeaknessBS))[dataset$WeaknessBS]
dataset$Type2Diabetes = as.numeric(levels(dataset$Type2Diabetes))[dataset$Type2Diabetes]
dataset$HeartAttack6M = as.numeric(levels(dataset$HeartAttack6M))[dataset$HeartAttack6M]
dataset$PeripheralArterialDiseases = as.numeric(levels(dataset$PeripheralArterialDiseases))[dataset$PeripheralArterialDiseases]
dataset$Smoking = as.numeric(levels(dataset$Smoking))[dataset$Smoking]
dataset$Asthma = as.numeric(levels(dataset$Asthma))[dataset$Asthma]
dataset$DGN = as.numeric(levels(dataset$DGN))[dataset$DGN]
dataset$PerformanceStatus = as.numeric(levels(dataset$PerformanceStatus))[dataset$PerformanceStatus]
dataset$SizeOfTumer = as.numeric(levels(dataset$SizeOfTumer))[dataset$SizeOfTumer]

# Encoding the features as factors for dataset1
dataset1$Risk1Yr = factor(dataset1$Risk1Yr,levels = c('TRUE','FALSE'),labels = c(1,0))
dataset1$PainBS = factor(dataset1$PainBS,levels = c('TRUE','FALSE'),labels = c(1,0))
dataset1$HaemoptysisBS = factor(datase1t$HaemoptysisBS,levels = c('TRUE','FALSE'),labels = c(1,0))
dataset1$DyspnoeaBS = factor(dataset1$DyspnoeaBS,levels = c('TRUE','FALSE'),labels = c(1,0))
dataset1$CoughBS = factor(dataset1$CoughBS,levels = c('TRUE','FALSE'),labels = c(1,0))
dataset1$WeaknessBS = factor(dataset1$WeaknessBS,levels = c('TRUE','FALSE'),labels = c(1,0))
dataset1$Type2Diabetes = factor(dataset1$Type2Diabetes,levels = c('TRUE','FALSE'),labels = c(1,0))
dataset1$HeartAttack6M = factor(dataset1$HeartAttack6M,levels = c('TRUE','FALSE'),labels = c(1,0))
dataset1$PeripheralArterialDiseases = factor(dataset1$PeripheralArterialDiseases,levels = c('TRUE','FALSE'),labels = c(1,0))
dataset1$Smoking = factor(dataset1$Smoking,levels = c('TRUE','FALSE'),labels = c(1,0))
dataset1$Asthma = factor(dataset1$Asthma,levels = c('TRUE','FALSE'),labels = c(1,0))
dataset1$DGN = factor(dataset1$DGN,levels = c('DGN1','DGN2','DGN3','DGN4','DGN5','DGN6','DGN8'),labels = c(1,2,3,4,5,6,8))
dataset1$PerformanceStatus = factor(dataset1$PerformanceStatus,levels = c('PRZ0','PRZ1','PRZ2'),labels = c(0,1,2))
dataset1$SizeOfTumer = factor(dataset1$SizeOfTumer,levels = c('OC11','OC12','OC13','OC14'),labels = c(11,12,13,14))

#features selections
risk <- dataset$Risk1Yr
risk1 <- dataset1$Risk1Yr
pvalue<-vector()
for (i in 1:16){
  ttestscore <- t.test(x=dataset[i], y= dataset$Risk1Yr)
  pvalue <- c(pvalue,ttestscore$p.value)
  if (pvalue[i] > 0.05){
    dataset <- dataset[-i]
    dataset1<- dataset1[-i]
  }
}
dataset$Risk1Yr <- risk
dataset1$Risk1Yr <- risk1
 
# features scaling
dataset[,12:14]=scale(dataset[,12:14])


summary(dataset)
glimpse(dataset)
skim(dataset)
dim(dataset)

# Splitting the dataset into the Training set and Test set for dataset and dataset1
set.seed(32312)
split = sample.split(dataset,SplitRatio = 0.85)
training_set = subset(dataset,split == TRUE)
test_set = subset(dataset,split == FALSE)
training_set1 = subset(dataset1,split == TRUE)
test_set1 = subset(dataset1,split == FALSE)




#Cross Validation For Naive Bayes
folds1 = createFolds(training_set1$Risk1Yr, k = 10)
cv1 = lapply(folds1, function(x) {
  training_fold = training_set1[-x, ]
  test_fold = training_set1[x, ]
  classifier= naiveBayes(x= training_fold[-15],y=training_fold$Risk1Yr)
  y_pred = predict(classifier,newdata =test_fold[-15])
  cm = table(test_fold[,15],y_pred)
  print(cm)
  accuracy = (cm[1,1] + cm[2,2]) / (cm[1,1] + cm[2,2] + cm[1,2] + cm[2,1])
  Sen = cm[1,1]  / (cm[1,1] + cm[1,2])
  Spec =1-(cm[2,1]  / (cm[2,1] + cm[2,2]))
  list <- list(accuracy,Sen,Spec)
  return(list)
})
accuracies1 <- vector(mode="character", length=10)
for (i in 1:10)
{
  accuracies1[i] <- c(cv1[[i]][1])
}
accuracy1 = mean(as.numeric(accuracies1))

sensitivities1 <- vector(mode="character", length=10)
for (i in 1:10)
{
  sensitivities1[i] <- c(cv1[[i]][2])
}
sens1 = mean(as.numeric(sensitivities1))

specificties1 <- vector(mode="character", length=10)
for (i in 1:10)
{
  specificties1[i] <- c(cv1[[i]][3])
}
spec1 = mean(as.numeric(specificties1))


#Cross Validation For KNN
folds2 = createFolds(training_set1$Risk1Yr, k = 10)
cv2 = lapply(folds2, function(x) {
  training_fold = training_set1[-x, ]
  test_fold = training_set1[x, ]
  y_pred = knn(train=training_fold[,-15],test=test_fold[,-15],cl=training_fold[,15],k=5)
  cm = table(test_fold[,15],y_pred)
  print(cm)
  accuracy = (cm[1,1] + cm[2,2]) / (cm[1,1] + cm[2,2] + cm[1,2] + cm[2,1])
  Sen =  cm[1,1]  / (cm[1,1] + cm[1,2])
  Spec =1-(cm[2,1]  / (cm[2,1] + cm[2,2])) 
  list <- list(accuracy,Sen,Spec)
  return(list)
})

accuracies2 <- vector(mode="character", length=10)
for (i in 1:10)
{
  accuracies2[i] <- c(cv2[[i]][1])
}
accuracy2 = mean(as.numeric(accuracies2))

sensitivities2 <- vector(mode="character", length=10)
for (i in 1:10)
{
  sensitivities2[i] <- c(cv2[[i]][2])
}
sens2 = mean(as.numeric(sensitivities2))

specificties2 <- vector(mode="character", length=10)
for (i in 1:10)
{
  specificties2[i] <- c(cv2[[i]][3])
}
spec2 = mean(as.numeric(specificties2))


# Cross validation For Logistic Regression using Bagging and Stratified KFold

#Function for Stratified KFold 
createFolds <- function(x,k){
  n <- nrow(x)
  x$folds <- rep(1:k,length.out = n)[sample(n,n)]
  x
}


# divide row size by 20, sample data 400 times 
length_divisor <- 20
accuracies3 <- vector()
Sen3 <- vector()
Spec3 <- vector()
acc3 <- vector(mode="character", length=10)
Se3 <- vector(mode="character", length=10)
Sp3 <- vector(mode="character", length=10)
for (m in 1:100){ 
  
  # using sample function without seed
  str_data <- ddply(training_set,.(training_set$Risk1Yr),createFolds,k = 10 )[-1]
  
  for (i in 1:10){
    
    testIndexes <- which(str_data$folds == i, arr.ind = TRUE)
    data_test <- str_data[testIndexes,][-16]
    data_train <- str_data[-testIndexes,][-16]
    classifier <- glm( Risk1Yr ~ .,
                       family = "binomial",
                       data = data_train  , maxit = 50)
    prob_pred = predict(classifier, type = 'response', newdata = data_test[-15])
    y_pred <- ifelse(prob_pred > 0.5, 1,0)
    y_pred = factor(y_pred,levels=c(0,1))
    cm =  table (data_test[,15] , y_pred)
    print(cm)
    acc3[i] <- c((cm[1,1] + cm[2,2]) / (cm[1,1] + cm[2,2] + cm[1,2] + cm[2,1]))
    Se3[i] <- c(cm[2,2]  / (cm[2,2] + cm[2,1]))
    Sp3[i] <- c(1-(cm[1,2]  / (cm[1,2] + cm[1,1])))
  }
}
accuracies3<-c(accuracies3,acc3)
Sen3<-c(Sen3,Se3)
Spec3<-c(Spec3,Sp3)

accuracy3 = mean(as.numeric(accuracies3))

sens3 = mean(as.numeric(Sen3))

spec3 = mean(as.numeric(Spec3))
