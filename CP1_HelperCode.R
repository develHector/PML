rm(list=ls())
getwd()

# install.packages("caret")
library(caret)

# install.packages("randomForest")
library(randomForest)

# install.packages("ipred")
library(ipred)

# install.packages("plyr")
library(plyr)

# Multicore Parallel Processing
#install.packages("parallel") 
library(parallel)

#install.packages("doParallel")
library(doParallel)

Cores <- parallel::detectCores() ;
registerDoParallel( cores = Cores )

###################################################################################################################################
# Read the Data
testing <- read.csv( "~/data/ml/pml-testing.csv", header = TRUE ) ;
training <- read.csv( "~/data/ml/pml-training.csv", header = TRUE ) ;
names(training)

###################################################################################################################################
# Remove columns with all NA values

na.cols <- NULL ;
for( i in 1:dim( training )[2] )
{  
  sum.nas <- sum( is.na( training[,i] ) ) ;
  if( sum.nas != 0 )
  {
    na.cols <- c( na.cols, i ) ;
  }  
}

training.complete <- training[ , -na.cols ] ;
str(training.complete)

for( i in 1:(dim( training.complete )[2]-1) )
{ 
  if ( class( training.complete[ , i ] ) == "factor" )
  {
    training.complete[ , i ] <- as.numeric( as.character( training.complete[ , i ] ) ) ;
  }
}
str(training.complete)


na.cols <- NULL ;
for( i in 1:dim( training.complete )[2] )
{  
  sum.nas <- sum( is.na( training.complete[,i] ) ) ;
  if( sum.nas != 0 )
  {
    na.cols <- c( na.cols, i ) ;
  }  
}

training.complete <- training.complete[ , -na.cols ] ;
str(training.complete)


###################################################################################################################################
# Get rid of Newar-zero variance columns

# training.complete <- training[, !caret::nearZeroVar( training, saveMetrics=TRUE )$nzv ]

###################################################################################################################################
# Since simple counter and consecutive Ids amd Timestamps had nothing to do, im getting noise about it since it gets 100% relevance
training.complete$X = NULL ;
training.complete$raw_timestamp_part_1 = NULL ;
training.complete$raw_timestamp_part_2 = NULL ;
training.complete$cvtd_timestamp = NULL ;
training.complete$num_window = NULL ;

###################################################################################################################################
# user_name as numeric
# user_id_table <- data.frame( user_id = c(1:length(levels( training.complete$user_name ))), user_name = levels( training.complete$user_name ) )
# training.complete$user_name <- merge( training.complete, user_id_table, by = "user_name", all = TRUE )$user_id

###################################################################################################################################
# Partition data in order to gather the ML method suited for our needs
Partition0 <- caret::createDataPartition( y = training.complete$classe, p = 0.20, list = FALSE ) ;
training.Partition0 <- training.complete[ Partition0,  ] ;

##################################################################################################################################
# Get rid of highly correlated predictors
# My bad, ended up not being that relevant
descrCor <-  abs( cor( training.Partition0[ , -dim( training.Partition0 ) [2] ] ) )
diag( descrCor ) <- 0
which ( descrCor  > 0.95, arr.ind = TRUE )

###################################################################################################################################
# PCA
# My bad, ended up not being that relevant
preProcPCA <- caret::preProcess( training.Partition0[,-length(names(training.Partition0))], method="pca", thresh = 0.995  )
preProcPCA
trainPCA <- stats::predict( preProcPCA, training.Partition0[,-length(names(training.Partition0))] )
trainPCA

# Sounds great to get only the principals, but the accuracy gets drammatically reduced
modFit.rpart.PCA <- caret::train( training.Partition0$classe ~ ., method = "treebag", data = trainPCA, na.action = na.roughfix, verbose=FALSE ) 
modFit.rpart.PCA$results$Accuracy
# [1] 0.2866122 

###################################################################################################################################
# Failed methods
modFit.glm <- caret::train( classe ~ . , method = "glm", data = training.Partition0, na.action = na.roughfix, verbose=FALSE )
modFit.rpart <- caret::train( classe ~ ., method = "rpart", data = training.Partition0, na.action = na.roughfix, verbose=FALSE ) 
system.time( modFit.rpart <- caret::train( classe ~ ., method = "rpart2", data = training.Partition0, na.action = na.roughfix, verbose=FALSE ) )


###################################################################################################################################
# Linear Discriminant Analysis
# install.packages("MASS")
library(MASS)
system.time( modFit.rpart <- caret::train( classe ~ ., method = "lda", data = training.Partition0, na.action = na.roughfix, verbose=FALSE ) )
#   user  system elapsed 
# 102.755   2.385   1.268 
   
modFit.rpart$finalModel
modFit.rpart
cm <- predict( modFit.rpart, newdata = training.Partition0, na.action = na.roughfix )
confusionMatrix( training.Partition0$classe, cm  )

#               Accuracy : 1          
#                 95% CI : (0.9991, 1)
#    No Information Rate : 0.2842     
#    P-Value [Acc > NIR] : < 2.2e-16 

###################################################################################################
# Stochastic Gradient Boosting
# install.packages("splines")
library(splines)
# install.packages("gbm")
library(gbm)

system.time( modFit.rpart <- caret::train( classe ~ ., method = "gbm", data = training.Partition0, na.action = na.roughfix, verbose=FALSE ) ) 
#    user  system elapsed 
#    472.594   2.516  76.969 

modFit.rpart$finalModel
modFit.rpart
cm <- predict( modFit.rpart, newdata = training.Partition0, na.action = na.roughfix )
confusionMatrix( training.Partition0$classe, cm  )

#               Accuracy : 1          
#                 95% CI : (0.9991, 1)
#    No Information Rate : 0.2842     
#    P-Value [Acc > NIR] : < 2.2e-16  


###################################################################################################################################
# Random Forest
system.time( modFit.rpart <- caret::train( classe ~ ., method = "rf", data = training.Partition0, na.action = na.roughfix, verbose=FALSE ) )
#  user  system elapsed 
#  623.453   5.481  97.056 
    
modFit.rpart$finalModel
modFit.rpart
cm <- predict( modFit.rpart, newdata = training.Partition0, na.action = na.roughfix )
confusionMatrix( training.Partition0$classe, cm  )

#               Accuracy : 1          
#                 95% CI : (0.9991, 1)
#    No Information Rate : 0.2842     
#    P-Value [Acc > NIR] : < 2.2e-16  

###################################################################################################
# Bagged CART - This one I liked     !   ##########################################################
library(ipred)

system.time( modFit.rpart <- caret::train( classe ~ ., method = "treebag", data = training.Partition0, na.action = na.roughfix, verbose=FALSE ) )
#  user  system elapsed 
#  124.040   3.584   9.469 
    
modFit.rpart$finalModel
modFit.rpart
cm <- predict( modFit.rpart, newdata = training.Partition0, na.action = na.roughfix )
confusionMatrix( training.Partition0$classe, cm  )

#               Accuracy : 1          
#                 95% CI : (0.9991, 1)
#    No Information Rate : 0.2842     
#    P-Value [Acc > NIR] : < 2.2e-16  



###################################################################################################
###################################################################################################################################
# Ok, treebag has a nice balance between speed and accuracry, but it's needed to Partition data but this in k-folds to perform cross validation in order to review

# Let's first create each of the folds
training.TrainFolds <- caret::createFolds( y = training.complete$classe, k=10, returnTrain = TRUE )

resultsPredict <- data.frame();

for( i in 1:length( training.TrainFolds ) )
  { 
    # Even it will be a random fold, let's use this x-validation technique
    training.kfold <- training.complete[ training.TrainFolds[[i]], ] ;
    testing.kfold <- training.complete[ -training.TrainFolds[[i]], ] ;  
  
    message( paste( "Predicting k-fold", i) )
    modFit <- caret::train( classe ~ ., method = "treebag", data = training.kfold, na.action = na.roughfix, verbose=FALSE ) ;
    
    # Being this model fitting a lengthy process, it's better to save it on disk in order to retrieve them during the writing up
    saveRDS( modFit, file = paste( "~/data/ml/modFit", i, ".rds", sep="." ) );
        
    modPredict <- predict( modFit, newdata = testing.kfold, na.action = na.roughfix )
    cmtPredict <- confusionMatrix( testing.kfold$classe, modPredict  )  
    
    # The x-validation will portray a clear idea of how precise the model is between folds
    resultsPredict <- rbind( resultsPredict, cmtPredict$overall )
    colnames(resultsPredict) <- names(cmtPredict$overall)  
    
  }

Predicting k-fold 1
Predicting k-fold 2
Predicting k-fold 3
Predicting k-fold 4
Predicting k-fold 5
Predicting k-fold 6
Predicting k-fold 7
Predicting k-fold 8
Predicting k-fold 9
Predicting k-fold 10

resultsPredict
    Accuracy     Kappa AccuracyLower AccuracyUpper AccuracyNull AccuracyPValue McnemarPValue
1  0.9979613 0.9974214     0.9947883     0.9994442    0.2844037              0           NaN
2  0.9954105 0.9941954     0.9913057     0.9978993    0.2835288              0           NaN
3  0.9943906 0.9929057     0.9899855     0.9971966    0.2835288              0           NaN
4  0.9959225 0.9948426     0.9919816     0.9982380    0.2844037              0           NaN
5  0.9969403 0.9961300     0.9933524     0.9988764    0.2845487              0           NaN
6  0.9949058 0.9935555     0.9906515     0.9975545    0.2857871              0           NaN
7  0.9928681 0.9909841     0.9880627     0.9960956    0.2801834              0           NaN
8  0.9969450 0.9961369     0.9933625     0.9988781    0.2825866              0           NaN
9  0.9969419 0.9961323     0.9933558     0.9988769    0.2838940              0           NaN
10 0.9969435 0.9961341     0.9933591     0.9988775    0.2842588              0           NaN


###################################################################################################

******************************************************************************
  
modFitList <- list();
for( i in 1:length( training.TrainFolds ) )
{ 
  message( paste( "Opening predictor", i) )
  filename = paste( "~/data/ml/modFit", i, ".rds", sep="." )
  modFitList[i] <- readRDS( file = filename );
}



###################################################################################################
message( "Predicting Full Training Set" )
modFit <- caret::train( classe ~ ., method = "treebag", data = training.complete, na.action = na.roughfix, verbose=FALSE ) ;

# This is what made me get back
library(rpart)
caret::varImp( modFit )
    
# Being this model fitting a lengthy process, it's better to save it on disk in order to retrieve them during the writing up
saveRDS( modFit, file =  "~/data/ml/modFit.full.rds" );

*********************************************************

modFit  = readRDS( file =  "~/data/ml/modFit.full.rds" );

modPredict <- predict( modFit, newdata = training.complete, na.action = na.roughfix )
cmtPredict <- confusionMatrix( training.complete$classe, modPredict  )  


###################################################################################################################################
# Apply exactly same rules to testing

str(testing)

testing.complete <- data.frame( testing ) ;

for( i in 1:( dim( testing.complete )[2]) )
{ 
  if ( class( testing.complete[ , i ] ) == "logical" )
  {
    testing.complete[ , i ] <- as.numeric( testing.complete[ , i ] ) ;
  }
}
str(testing.complete)

modPredict <- predict( modFit, newdata = testing.complete, na.action = na.pass )
modPredict

# These are the results per test case, and after my submission, everyeone of them were (gladly) correct!
[1] B A B A A E D B A A B C B A E E A B B B
Levels: A B C D E

# As commanded by the instructions, generate the files and write them down to files to submit them
pml_write_files = function(x)
{
  n = length(x)
  for(i in 1:n)
  {
    filename = paste0( "~/data/ml/problem_id_", i, ".txt" ) ;
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}
answers <- as.character( modPredict ) 
pml_write_files( answers )



# Assignment manual generation
library(knitr)
knit2html("CP1_PracticalMachineLearning.Rmd")
setwd(".\\8. Machine Learning")
getwd()
    
