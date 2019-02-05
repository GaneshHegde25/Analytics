# Load all the packages required for the analysis
#library(dplyr) # Data Manipulation
#library(Amelia) # Missing Data: Missings Map
#library(ggplot2) # Visualization
#library(scales) # Visualization
#library(caTools) # Prediction: Splitting Data
#library(car) # Prediction: Checking Multicollinearity
#library(ROCR) # Prediction: ROC Curve
#library(e1071) # Prediction: SVM, Naive Bayes, Parameter Tuning
library(rpart) # Prediction: Decision Tree
library(rpart.plot) # Prediction: Decision Tree
library(randomForest) # Prediction: Random Forest
library(caret) # Prediction: k-Fold Cross Validation
#library(doSNOW) #Training in Parallel

#Read the data
read_titanic_data = read.csv("Titanic_data.csv")

# Checking missing values (missing values or empty values)
colSums(is.na(read_titanic_data)|read_titanic_data=='')

#Missing Embarked Data Imputation
#Get the mode
# Create the function.
getmode <- function(v) {
  uniqv <- unique(v)
  uniqv[which.max(tabulate(match(v, uniqv)))]
}
v = read_titanic_data$Embarked

result =getmode(v)

#relacing Na's with mode of the column
read_titanic_data$Embarked[read_titanic_data$Embarked == ""] = result

#Let's remove all those columns from dataset which are not relevant
feature= c("Survived", "Pclass", "Sex", "Age", "SibSp","Parch", "Fare", "Embarked")

read_titanic_data =read_titanic_data[,feature]




#Let's impute missing ages based on PCclass
impute.age <- function(age,class){
  vector <- age
  for (i in 1:length(age)){
    if (is.na(age[i])){
      if (class[i] == 1){
        vector[i] <- round(mean(filter(read_titanic_data,Pclass==1)$Age, na.rm=TRUE),0)
      }else if (class[i] == 2){
        vector[i] <- round(mean(filter(read_titanic_data,Pclass==2)$Age, na.rm=TRUE),0)
      }else{
        vector[i] <- round(mean(filter(read_titanic_data,Pclass==3)$Age, na.rm=TRUE),0)
      }
    }else{
      vector[i]<-age[i]
    }
  }
  return(vector)
}
imputed.age <- impute.age(read_titanic_data$Age,read_titanic_data$Pclass)
read_titanic_data$Age <- imputed.age




# Checking missing values
#colSums(is.na(read_titanic_data)|read_titanic_data=='') #No missing data found

#Encoding the categorical features as factors

read_titanic_data$Survived <- as.factor(read_titanic_data$Survived)
read_titanic_data$Pclass <- as.factor(read_titanic_data$Pclass)
read_titanic_data$Sex <- as.factor(read_titanic_data$Sex)
read_titanic_data$Embarked <- as.factor(read_titanic_data$Embarked)
read_titanic_data$MissingAge <- as.factor(read_titanic_data$MissingAge)


#Splitting data into test and train data using Caret package 70% of data into training set and rest 30% into testing set
set.seed(32984)
indexes =createDataPartition(read_titanic_data$Survived,times = 1,p=0.7,list = FALSE)

#Creating Train and test data
training_data= read_titanic_data[indexes,]
test_data= read_titanic_data[-indexes,]

#Now we will verify proportions of our labels in data set and match it with original file
prop.table(table(read_titanic_data$Survived))
prop.table(table(training_data$Survived))
prop.table(table(test_data$Survived))


#Prediction using Decision Tree

# Fitting Decision Tree Classification Model to the Training set
classifier = rpart(Survived ~ ., data = training_data, method = 'class')

# Tree Visualization
rpart.plot(classifier, extra=4)


# Predicting the Validation set results
y_pred = predict(classifier, newdata = test_data[,-which(names(test_data)=="Survived")], type='class')

# Checking the prediction accuracy
table(test_data$Survived, y_pred) # Confusion matrix

#Error rate
error <- mean(test_data$Survived != y_pred) # Misclassification error
paste('Accuracy',round(1-error,4))
