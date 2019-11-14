library(ggplot2)
library(caTools)
library(e1071)
library(ElemStatLearn)
library(class)
library(caret)
library(glmnet)
library(dplyr)
library(skimr)

dataset <- read.csv(file="ThoraricSurgery.csv")
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

summary(dataset$Risk1Yr)
glimpse(dataset)
skim(dataset$Risk1Yr)
dim(dataset)

set.seed(333)
split = sample.split(dataset$Risk1Yr,SplitRatio = 0.8)
training_set = subset(dataset,split == TRUE)
test_set = subset(dataset,split == FALSE)


training_set[,14:16]=scale(training_set[,14:16])
test_set[,14:16]=scale(test_set[,14:16])

folds1 = createFolds(training_set$Risk1Yr, k = 10)
cv1 = lapply(folds1, function(x) {
  training_fold = training_set[-x, ]
  test_fold = training_set[x, ]
  classifier= naiveBayes(x= training_fold[-17],y=training_fold$Risk1Yr)
  y_pred = predict(classifier,newdata =test_fold[-17])
  cm = table(test_fold[,17],y_pred)
  print(cm)
  accuracy = (cm[1,1] + cm[2,2]) / (cm[1,1] + cm[2,2] + cm[1,2] + cm[2,1])
  Sen = cm[1,1]  / (cm[1,1] + cm[2,1])
  Spec = cm[2,2]  / (cm[1,2] + cm[2,2])
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

folds2 = createFolds(training_set$Risk1Yr, k = 10)
cv2 = lapply(folds2, function(x) {
  training_fold = training_set[-x, ]
  test_fold = training_set[x, ]
  y_pred = knn(train=training_fold[,-17],test=test_fold[,-17],cl=training_fold[,17],k=5)
  cm = table(test_fold[,17],y_pred)
  print(cm)
  accuracy = (cm[1,1] + cm[2,2]) / (cm[1,1] + cm[2,2] + cm[1,2] + cm[2,1])
  Sen = cm[1,1]  / (cm[1,1] + cm[2,1])
  Spec = cm[2,2]  / (cm[1,2] + cm[2,2])
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




folds3 = createFolds(training_set$Risk1Yr, k = 10)
cv3 = lapply(folds3, function(x) {
  training_fold = training_set[-x, ]
  test_fold = training_set[x, ]
  classifier = glm(formula = Risk1Yr ~ .,
                   family = binomial,
                   data = training_fold)
  
  prob_pred = predict(classifier, type = 'response', newdata = test_fold[-17])
  y_pred = ifelse(prob_pred > 0.5, 1, 0)
  y_pred = as.integer(y_pred)
  y_pred = factor(y_pred,levels=c(1,0))
  cm = table(test_fold[,17], y_pred,useNA = "always" )
  print(cm)
  accuracy = (cm[1,1] + cm[2,2]) / (cm[1,1] + cm[2,2] + cm[1,2] + cm[2,1])
  Sen = cm[1,1]  / (cm[1,1] + cm[2,1])
  Spec = cm[2,2]  / (cm[1,2] + cm[2,2])
  list <- list(accuracy,Sen,Spec)
  return(list)
})

accuracies3 <- vector(mode="character", length=10)
for (i in 1:10)
{
  accuracies3[i] <- c(cv3[[i]][1])
}
accuracy3 = mean(as.numeric(accuracies3))

sensitivities3 <- vector(mode="character", length=10)
for (i in 1:10)
{
  sensitivities3[i] <- c(cv3[[i]][2])
}
sens3 = mean(as.numeric(sensitivities3))

specificties3 <- vector(mode="character", length=10)
for (i in 1:10)
{
  specificties3[i] <- c(cv3[[i]][3])
}
spec3 = mean(as.numeric(specificties3))
  
