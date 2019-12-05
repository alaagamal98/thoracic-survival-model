library(caTools) 
library(e1071) 
library(ElemStatLearn)
library(class)
library(caret) 
library(glmnet)
library(plyr)
library(dplyr) 
library(ipred)
library(skimr)
library(foreach)
library(doParallel)


# Importing the dataset
dataset <- read.csv(file="ThoraricSurgery.csv")
datasetplot <- read.csv(file="ThoraricSurgery.csv")

# Encoding the features as factors
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

# Encoding the features as factors
datasetplot$Risk1Yr = factor(datasetplot$Risk1Yr,levels = c('TRUE','FALSE'),labels = c(1,0))
datasetplot$PainBS = factor(datasetplot$PainBS,levels = c('TRUE','FALSE'),labels = c(1,0))
datasetplot$HaemoptysisBS = factor(datasetplot$HaemoptysisBS,levels = c('TRUE','FALSE'),labels = c(1,0))
datasetplot$DyspnoeaBS = factor(datasetplot$DyspnoeaBS,levels = c('TRUE','FALSE'),labels = c(1,0))
datasetplot$CoughBS = factor(datasetplot$CoughBS,levels = c('TRUE','FALSE'),labels = c(1,0))
datasetplot$WeaknessBS = factor(datasetplot$WeaknessBS,levels = c('TRUE','FALSE'),labels = c(1,0))
datasetplot$Type2Diabetes = factor(datasetplot$Type2Diabetes,levels = c('TRUE','FALSE'),labels = c(1,0))
datasetplot$HeartAttack6M = factor(datasetplot$HeartAttack6M,levels = c('TRUE','FALSE'),labels = c(1,0))
datasetplot$PeripheralArterialDiseases = factor(datasetplot$PeripheralArterialDiseases,levels = c('TRUE','FALSE'),labels = c(1,0))
datasetplot$Smoking = factor(datasetplot$Smoking,levels = c('TRUE','FALSE'),labels = c(1,0))
datasetplot$Asthma = factor(datasetplot$Asthma,levels = c('TRUE','FALSE'),labels = c(1,0))
datasetplot$DGN = factor(datasetplot$DGN,levels = c('DGN1','DGN2','DGN3','DGN4','DGN5','DGN6','DGN8'),labels = c(1,2,3,4,5,6,8))
datasetplot$PerformanceStatus = factor(datasetplot$PerformanceStatus,levels = c('PRZ0','PRZ1','PRZ2'),labels = c(0,1,2))
datasetplot$SizeOfTumer = factor(datasetplot$SizeOfTumer,levels = c('OC11','OC12','OC13','OC14'),labels = c(11,12,13,14))


# Feature Scaling
dataset[,14:16]=scale(dataset[,14:16])


summary(dataset)
glimpse(dataset)
skim(dataset)
dim(dataset)

# Splitting the dataset into the Training set and Test set
set.seed(32312)
split = sample.split(dataset,SplitRatio = 0.85)
training_set = subset(dataset,split == TRUE)
test_set = subset(dataset,split == FALSE)


#Cross Validation For Naive Bayes
folds1 = createFolds(training_set$Risk1Yr, k = 10)
cv1 = lapply(folds1, function(x) {
  training_fold = training_set[-x, ]
  test_fold = training_set[x, ]
  classifier= naiveBayes(x= training_fold[-17],y=training_fold$Risk1Yr)
  y_pred = predict(classifier,newdata =test_fold[-17])
  cm = table(test_fold[,17],y_pred)
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
folds2 = createFolds(training_set$Risk1Yr, k = 10)
cv2 = lapply(folds2, function(x) {
  training_fold = training_set[-x, ]
  test_fold = training_set[x, ]
  y_pred = knn(train=training_fold[,-17],test=test_fold[,-17],cl=training_fold[,17],k=5)
  cm = table(test_fold[,17],y_pred)
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




#Converting Factors to Numeric
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


training_set = subset(dataset,split == TRUE)
test_set = subset(dataset,split == FALSE)




#Converting Factors to Numeric
datasetplot$Risk1Yr = as.numeric(levels(datasetplot$Risk1Yr))[datasetplot$Risk1Yr]
datasetplot$PainBS = as.numeric(levels(datasetplot$PainBS))[datasetplot$PainBS]
datasetplot$HaemoptysisBS = as.numeric(levels(datasetplot$HaemoptysisBS))[datasetplot$HaemoptysisBS]
datasetplot$DyspnoeaBS = as.numeric(levels(datasetplot$DyspnoeaBS))[datasetplot$DyspnoeaBS]
datasetplot$CoughBS = as.numeric(levels(datasetplot$CoughBS))[datasetplot$CoughBS]
datasetplot$WeaknessBS = as.numeric(levels(datasetplot$WeaknessBS))[datasetplot$WeaknessBS]
datasetplot$Type2Diabetes = as.numeric(levels(datasetplot$Type2Diabetes))[datasetplot$Type2Diabetes]
datasetplot$HeartAttack6M = as.numeric(levels(datasetplot$HeartAttack6M))[datasetplot$HeartAttack6M]
datasetplot$PeripheralArterialDiseases = as.numeric(levels(datasetplot$PeripheralArterialDiseases))[datasetplot$PeripheralArterialDiseases]
datasetplot$Smoking = as.numeric(levels(datasetplot$Smoking))[datasetplot$Smoking]
datasetplot$Asthma = as.numeric(levels(datasetplot$Asthma))[datasetplot$Asthma]
datasetplot$DGN = as.numeric(levels(datasetplot$DGN))[datasetplot$DGN]
datasetplot$PerformanceStatus = as.numeric(levels(datasetplot$PerformanceStatus))[datasetplot$PerformanceStatus]
datasetplot$SizeOfTumer = as.numeric(levels(datasetplot$SizeOfTumer))[datasetplot$SizeOfTumer]


training_setplot = subset(datasetplot,split == TRUE)
test_setplot = subset(datasetplot,split == FALSE)



createFolds <- function(x,k){
  n <- nrow(x)
  x$folds <- rep(1:k,length.out = n)[sample(n,n)]
  x
}

# divide row size by 20, sample data 100 times 
length_divisor <- 20
accuracies3 <- vector()
Sen3 <- vector()
Spec3 <- vector()
acc3 <- vector(mode="character", length=10)
Se3 <- vector(mode="character", length=10)
Sp3 <- vector(mode="character", length=10)

#Cross Validation For Logisitic Regression
for (m in 1:100){ 
  # using sample function without seed
  str_data <- ddply(training_set,.(training_set$Risk1Yr),createFolds,k = 10 )[-1]
  
  for (i in 1:10){
    
    testIndexes <- which(str_data$folds == i, arr.ind = TRUE)
    data_test <- str_data[testIndexes,][-18]
    data_train <- str_data[-testIndexes,][-18]
    classifier <- glm( Risk1Yr ~ .,
                       family = "binomial",
                       data = data_train  , maxit = 50)
    prob_pred = predict(classifier, type = 'response', newdata = data_test[-17])
    y_pred <- ifelse(prob_pred > 0.5, 1,0)
    y_pred = factor(y_pred,levels=c(0,1))
    cm =  table (data_test[,17] , y_pred)
    
    print(cm)
    acc3[i] <- c((cm[1,1] + cm[2,2]) / (cm[1,1] + cm[2,2] + cm[1,2] + cm[2,1]))
    Se3[i] <- c(cm[2,2]  / (cm[2,2] + cm[2,1]))
    Sp3[i] <- c(1-(cm[1,2]  / (cm[1,2] + cm[1,1])))
  }
  accuracies3<-c(accuracies3,acc3)
 Sen3<-c(Sen3,Se3)
 Spec3<-c(Spec3,Sp3)

}

accuracy3 = mean(as.numeric(accuracies3))
sens3 = mean(as.numeric(Sen3))
spec3 = mean(as.numeric(Spec3))

ggplot(data=datasetplot, aes(x=Age, y=SizeOfTumer))+
  geom_point(aes(color=Risk1Yr))

prob_pred = predict(classifier, type = 'response', newdata = test_setplot[-17])
test_setplot$prob_pred <-prob_pred

ggplot(test_setplot, aes(x=SizeOfTumer, y=prob_pred))+
  geom_point() +    geom_smooth(method='lm', formula= y~x)

ggplot(test_setplot, aes(x=Age, y=prob_pred))+
  geom_point() +    geom_smooth(method='lm', formula= y~x)

