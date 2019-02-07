#Load required packages
library(dplyr) 
library(Amelia) 
library(scales) 
library(caTools) 
library(e1071) 
library(rpart)
library(rpart.plot) 
library(randomForest) 
library(caret)

#Read the data
read_titanic_data = read.csv("Titanic_data.csv")
View(read_titanic_data)

#============================================================================
##Data Wrangling
#============================================================================

# Checking missing values (missing values or empty values)
colSums(is.na(read_titanic_data)|read_titanic_data=='')

#Missing Embarked Data Imputation
#Get the mode Create the function.
getmode <- function(v) {
  uniqv <- unique(v)
  uniqv[which.max(tabulate(match(v, uniqv)))]
}
v = read_titanic_data$Embarked

result =getmode(v)

read_titanic_data$Embarked[read_titanic_data$Embarked == ""] = result

#Let's impute missing ages based on PCclass
# First, transform all feature to dummy variables.
dummy.vars <- dummyVars(~ ., data = read_titanic_data[, -1])
train.dummy <- predict(dummy.vars, read_titanic_data[, -1])
View(train.dummy)

# Now, impute!
pre.process <- preProcess(train.dummy, method = "bagImpute")
imputed.data <- predict(pre.process, train.dummy)
View(imputed.data)

read_titanic_data$Age <- imputed.data[, 6]
View(read_titanic_data)

# Checking missing values
colSums(is.na(read_titanic_data)|read_titanic_data=='')

#===========================================================================
#Feature Engineering
#===========================================================================

# Add a feature for family size.
read_titanic_data$FamilySize <- 1 + read_titanic_data$SibSp + read_titanic_data$Parch

# Adding a new feature Title
#Grab passenger title from passenger name
read_titanic_data$Title <- gsub("^.*, (.*?)\\..*$", "\\1", read_titanic_data$Name)

# Frequency of each title by sex
table(read_titanic_data$Sex, read_titanic_data$Title)

# There are so many categories let's combine few categories and put less frequent categories as other
read_titanic_data$Title[read_titanic_data$Title == 'Mlle' | read_titanic_data$Title == 'Ms'] <- 'Miss' 
read_titanic_data$Title[read_titanic_data$Title == 'Mme']  <- 'Mrs' 

Other <- c('Dona', 'Dr', 'Lady', 'the Countess','Capt', 'Col', 'Don', 'Jonkheer', 'Major', 'Rev', 'Sir')
read_titanic_data$Title[read_titanic_data$Title %in% Other]  <- 'Other'

# Let's check title 
table(read_titanic_data$Sex, read_titanic_data$Title)
#Let's check data now
View(read_titanic_data)

#Now we will keep only relevant data attributes in our data
feature= c("Survived", "Pclass", "Sex", "Age", "SibSp","Parch", "Fare", "Embarked","FamilySize","Title")
read_titanic_data= read_titanic_data[,feature]
View(read_titanic_data)

#==================================================================================
#Data preparation
#==================================================================================

#Encoding the categorical features as factors

read_titanic_data$Survived <- as.factor(read_titanic_data$Survived)
read_titanic_data$Pclass <- as.factor(read_titanic_data$Pclass)
read_titanic_data$Sex <- as.factor(read_titanic_data$Sex)
read_titanic_data$Embarked <- as.factor(read_titanic_data$Embarked)
read_titanic_data$Title = factor(read_titanic_data$Title)
#read_titanic_data$FamilySize = factor(read_titanic_data$FamilySize, levels=c("Single","Small","Large"))


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


#=================================================================================
#Modelling using Decision Tree
#=================================================================================
# Fitting Decision Tree Classification Model to the Training set
classifier = rpart(Survived ~ ., data = training_data, method = 'class')

# Tree Visualization
rpart.plot(classifier, extra=4)


# Predicting the test set results
y_pred = predict(classifier, newdata = test_data[,-which(names(test_data)=="Survived")], type='class')

# Checking the prediction accuracy
table(test_data$Survived, y_pred) # Confusion matrix

#Error rate
error <- mean(test_data$Survived != y_pred) # Misclassification error
paste('Accuracy',round(1-error,4))

#=================================================================================
#Modelling using Random Forest
#=================================================================================

# Fitting Random Forest Classification to the Training set
set.seed(432)
classifier_R = randomForest(Survived ~ ., data = training_data)
# Choosing the number of trees
plot(classifier_R)

# Predicting the test set results
RF_pred = predict(classifier_R ,newdata = test_data[,-which(names(test_data)=="Survived")])

# Checking the prediction accuracy
table(test_data$Survived, RF_pred)

#Error rate
error <- mean(test_data$Survived != RF_pred) # Misclassification error
paste('Accuracy',round(1-error,4))

#=================================================================================
#Modelling using SVM
#=================================================================================

# Checking the variance of numeric features
paste('Age variance: ',var(training_data$Age),', SibSp variance: ',var(training_data$SibSp),', Parch variance: ',var(training_data$Parch),',
      Fare variance: ',var(training_data$Fare))

# Feature Scaling - use scale() to standardize the feature columns
standardized.train = cbind(select(training_data, Survived, Pclass, Sex, SibSp, Parch, Embarked, Title, FamilySize,Age), Fare = scale(training_data$Fare))
paste(', Fare variance: ',var(standardized.train$Fare))

paste('Age variance: ',var(test_data$Age),', SibSp variance: ',var(test_data$SibSp),', Parch variance: ',var(test_data$Parch),',
      Fare variance: ',var(test_data$Fare))

standardized.test = cbind(select(test_data, Survived, Pclass, Sex, SibSp, Parch, Embarked, Title, FamilySize,Age),  Fare = scale(test_data$Fare))
paste(', Fare variance: ',var(standardized.test$Fare))


# Fitting Linear SVM to the Training set
classifier = svm(Survived ~ .,
                 data = standardized.train,
                 type = 'C-classification',
                 kernel = 'linear')

# Predicting the Validation set results
Svm_pred = predict(classifier, newdata = standardized.test[,-which(names(standardized.test)=="Survived")])

# Checking the prediction accuracy
table(test_data$Survived, Svm_pred) # Confusion matrix

error <- mean(test_data$Survived != Svm_pred) # Misclassification error
paste('Accuracy',round(1-error,4))
