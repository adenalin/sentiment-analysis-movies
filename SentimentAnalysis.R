# PRESENTATION LINK: 
# http://rpubs.com/adena/sentiment1
# http://rpubs.com/adena/sentiment2

# set working directory to wherever file 'Sentiment Analysis' is located
currentwd <- setwd("/Users/Adena/Desktop/Sentiment Analysis/")

########
'Using lexicon to count +/- words'
########

library(RTextTools)
library(plyr)
library(stringr)
library(readr)

# Words that are categorized into positive/negative sentiment
# Source: http://www.cs.uic.edu/~liub/FBS/opinion-lexicon-English.rar
# Added +/- words from 'http://member.tokoha-u.ac.jp/~dixonfdm/Writing%20Topics%20htm/Movie%20Review%20Folder/movie_descrip_vocab.htm'
# that are not in lexicons
positiveWords <- scan("lexicon/positive-words.txt",
                  what = 'character')
positiveWords[2007:2015] <- c('comical',
                   'uproarious',
                   'original',
                   'absorbing',
                   'riveting',
                   'surprising',
                   'dazzling',
                   'thought-provoking', 
                   'unpretentious')
negativeWords <- scan("lexicon/negative-words.txt",
                    what = 'character')
negativeWords[4784:4790] <- c('second-rate',
                   'third-rate',
                   'juvenile', 
                   'ordinary',
                   'predictable',
                   'uninteresting',
                   'outdated')

# Function to process sentences - break down into individual words for
# sentiment analysis by comparing to lexicon
sentimentFunction <- function(sentences, positiveWords, negativeWords, .progress='none') {
  finalScore <- matrix('', 0, 3)
  scores <- laply(sentences, function(sentence, positiveWords, negativeWords) {
    compareSentence <- sentence
    
    # split into words
    word.list <- str_split(sentence, '\\s+')
    words <- unlist(word.list)
    
    # compare our words to the lexicon of +/- terms
    posMatches <- match(words, positiveWords)
    negMatches <- match(words, negativeWords)
    
    # find total number of words in each category
    posMatches <- sum(!is.na(posMatches))
    negMatches <- sum(!is.na(negMatches))
    
    score <- c(posMatches, negMatches)
    
    # add row to scores table
    new_row <- c(compareSentence, score)
    finalScore <- rbind(finalScore, new_row)
    return(finalScore)
}, positiveWords, negativeWords)
return(scores)
}

# import training data
setwd(paste(currentwd, "/postrain", sep=""))
pos <- list.files(path = setwd(paste(currentwd, "/postrain", sep="")), pattern = "*.txt")
positiveReviews <- lapply(pos, read_file)
# count number of periods in each review
positivePeriods <- lapply(gregexpr("[.]", positiveReviews), 
                          function(x) ifelse(x[[1]] > 0, 
                                             length(x), 
                                             as.integer(0)))
# count number of exclamation marks
positiveExclamations <- lapply(gregexpr("[!]", positiveReviews), 
                                function(x) ifelse(x[[1]] > 0, 
                                                   length(x), 
                                                   as.integer(0)))
# count number of question marks
positiveQuestions <- lapply(gregexpr("[?]", positiveReviews), 
                            function(x) ifelse(x[[1]] > 0, 
                                               length(x), 
                                               as.integer(0)))
# remove punctuation and digits for word processing
positiveReviews <- lapply(positiveReviews, gsub, pattern = '[[:punct:],\n]', replacement = '')
positiveReviews <- lapply(positiveReviews, gsub, pattern = '\\d+', replacement = '')

setwd(paste(currentwd, "/negtrain", sep=""))
neg <- list.files(path = setwd(paste(currentwd, "/negtrain", sep="")), pattern = "*.txt")
negativeReviews <- lapply(neg, read_file)
negativePeriods <- lapply(gregexpr("[.]", negativeReviews), 
                          function(x) ifelse(x[[1]] > 0, 
                                             length(x), 
                                             as.integer(0)))
negativeExclamations <- lapply(gregexpr("[!]", negativeReviews), 
                               function(x) ifelse(x[[1]] > 0, 
                                                  length(x), 
                                                  as.integer(0)))
negativeQuestions <- lapply(gregexpr("[?]", negativeReviews), 
                            function(x) ifelse(x[[1]] > 0, 
                                               length(x), 
                                               as.integer(0)))
negativeReviews <- lapply(negativeReviews, gsub, pattern = '[[:punct:],\n]', replacement = '')
negativeReviews <- lapply(negativeReviews, gsub, pattern = '\\d+', replacement = '')

# data frame of positive/negative sentences with scores
# this compares words from training data to lexicon to count number of +/- words
posResult <- as.data.frame(sentimentFunction(positiveReviews, positiveWords, negativeWords))
negResult <- as.data.frame(sentimentFunction(negativeReviews, positiveWords, negativeWords))
posResult <- cbind(posResult, 'positive')
colnames(posResult) <- c('sentence','positive', 'negative','sentiment')
negResult <- cbind(negResult, 'negative')
colnames(negResult) <- c('sentence', 'positive', 'negative', 'sentiment')
functionResults <- rbind(posResult, negResult)
# end up with data frame that lists the number of +/- words in each review

#######################
'RUN ONLY IF NOT USING RF FOR CLASSIFICATION'
# formula: more positive than negative words = positive sentiment

functionResults$predicted <- ifelse(functionResults$positive >= functionResults$negative, 
                                    'positive', 
                                    'negative')

# test on (compare with) original training data
# overall accuracy = 70%
recall_accuracy(functionResults$sentiment, functionResults$predicted)
# positive sentiments accuracy = 67%
recall_accuracy(functionResults$sentiment[1:1000], functionResults$predicted[1:1000])
# negative sentiments accuracy = 73%
recall_accuracy(functionResults$sentiment[1001:2000], functionResults$predicted[1001:2000])
#######################

# notes on 2nd iteration:
# figured out why data frame was not correctly labelling some "positive" reviews as "positive", fixed that
# by forcing numbers into characters before setting as.numeric; also set neutral (ie. # pos words = # number neg words)
# as 'positive', it was set to 'negative' before.
# Results (seen above): increased overall accuracy by 69%, positive accuracy by 9%, and negative accuracy by 1%

# next, I will try putting in the positive vs. negative words data into classification models to see
# if they will do a better job at prediction than my simple model. 

# There are more than double the number of negative words vs positive words, could be causing disbalance
# as to why positive accuracy is slightly lower than negative accuracy. To control for this, the
# number of +/- words in each review will be divided by the total number of +/- words in the lexicon.

#####
'ADD NEW FEATURES'
#####

functionResults$positive <- as.numeric(as.character(functionResults$positive))
functionResults$negative <- as.numeric(as.character(functionResults$negative))

# Create two new columns for proportion of +/- words (ie. + words counted / total # of + words)
functionResults$prop_pos <- functionResults$positive/(length(positiveWords)) # number of + words in lexicon
functionResults$prop_neg <- functionResults$negative/(length(negativeWords)) # number of _ words in lexicon

# add new column for number of words in each review to indicate length of review
wordcount <- function(str) {
  sapply(gregexpr("\\b\\W+\\b", str, perl=TRUE), function(x) sum(x>0) ) + 1 
}
functionResults$num_words <- wordcount(functionResults$sentence)

# add feature: difference between positive and negative words
functionResults$difference <- functionResults$positive - functionResults$negative

# add feature: proportion of positive to negative words
functionResults$prop_pos_neg <- ifelse(functionResults$negative == 0, 
                                       100, 
                                       functionResults$positive / functionResults$negative)

# add feature: proportion of positive or negative words to total words
functionResults$pos_totalwords <- functionResults$positive / functionResults$num_words
functionResults$neg_totalwords <- functionResults$negative / functionResults$num_words

# add feature: add the number of punctuation counted at beginning for each review
functionResults$periods[1:1000] <- positivePeriods
functionResults$periods[1001:2000] <- negativePeriods
functionResults$exclamation_marks[1:1000] <- positiveExclamations
functionResults$exclamation_marks[1001:2000] <- negativeExclamations
functionResults$q_marks[1:1000] <- positiveQuestions
functionResults$q_marks[1001:2000] <- negativeQuestions

# make sure the values for the punctuation columns are numbers
functionResults$periods <- as.numeric(as.character(functionResults$periods))
functionResults$exclamation_marks <- as.numeric(as.character(functionResults$exclamation_marks))
functionResults$q_marks <- as.numeric(as.character(functionResults$q_marks))

# remove 'sentence' column from matrix as it will not be needed for RF; create new matrix
sentimentMatrix <- functionResults
sentimentMatrix$sentence <- NULL

# use random forest & decision tree for classification
library(rpart)
rfAnalysis <- rpart(sentiment ~ .,
                           data = sentimentMatrix)
trainResults <- predict(rfAnalysis, sentimentMatrix)
trainResults <- data.frame(trainResults)
trainResults$sentiment <- ifelse(trainResults$positive > trainResults$negative, 'positive', 'negative')
# see accuracy for RF
recall_accuracy(sentimentMatrix$sentiment, trainResults$sentiment)
# note: after adding additional features, accuracy is 100% with random forests
# note: accuracy is 70.55% with decision tree

# plot decision tree
library(rpart.plot)
library(rattle)
fancyRpartPlot(rfAnalysis)
prp(rfAnalysis)

#######
'PREPARE TEST DATA'
#######

# Preprocess test data in same manner as training data; ie. create features
setwd(paste(currentwd, "/postest", sep=""))
pTest <- list.files(path = setwd(paste(currentwd, "/postest", sep="")), pattern = "*.txt")
positiveTest <- lapply(pTest, read_file)
posP <- lapply(gregexpr("[.]", positiveTest), 
                          function(x) ifelse(x[[1]] > 0, 
                                             length(x), 
                                             as.integer(0)))
# count number of exclamation marks
posE <- lapply(gregexpr("[!]", positiveTest), 
                               function(x) ifelse(x[[1]] > 0, 
                                                  length(x), 
                                                  as.integer(0)))
# count number of question marks
posQ <- lapply(gregexpr("[?]", positiveTest), 
                            function(x) ifelse(x[[1]] > 0, 
                                               length(x), 
                                               as.integer(0)))
positiveTest <- lapply(positiveTest, gsub, pattern = '[\n]', replacement = '')
positiveTest <- lapply(positiveTest, gsub, pattern = '\\d+', replacement = '')
positiveTest <- tolower(positiveTest)

setwd(paste(currentwd, "/negtest", sep=""))
nTest <- list.files(path = setwd(paste(currentwd, "/negtest", sep="")), pattern = "*.txt")
negativeTest <- lapply(nTest, read_file)
negP <- lapply(gregexpr("[.]", negativeTest), 
                          function(x) ifelse(x[[1]] > 0, 
                                             length(x), 
                                             as.integer(0)))
# count number of exclamation marks
negE <- lapply(gregexpr("[!]", negativeTest), 
                               function(x) ifelse(x[[1]] > 0, 
                                                  length(x), 
                                                  as.integer(0)))
# count number of question marks
negQ <- lapply(gregexpr("[?]", negativeTest), 
                            function(x) ifelse(x[[1]] > 0, 
                                               length(x), 
                                               as.integer(0)))
negativeTest <- lapply(negativeTest, gsub, pattern = '[\n]', replacement = '')
negativeTest <- lapply(negativeTest, gsub, pattern = '\\d+', replacement = '')
negativeTest <- tolower(negativeTest)

# consolidate +/- reviews into one large data frame (test data)
positiveTest <- cbind(positiveTest, 'positive')
negativeTest <- cbind(negativeTest, 'negative')
testData <- rbind(positiveTest, negativeTest)
posTest <- as.data.frame(sentimentFunction(positiveTest[,1], positiveWords, negativeWords))
negTest <- as.data.frame(sentimentFunction(negativeTest[,1], positiveWords, negativeWords))
posTest <- cbind(posTest, 'positive')
colnames(posTest) <- c('sentence','positive', 'negative','sentiment')
negTest <- cbind(negTest, 'negative')
colnames(negTest) <- c('sentence', 'positive', 'negative', 'sentiment')
testData <- rbind(posTest, negTest)
testData$positive <- as.numeric(as.character(testData$positive))
testData$negative <- as.numeric(as.character(testData$negative))

# create features
testData$prop_pos <- testData$positive/(length(positiveWords)) # number of + words in lexicon
testData$prop_neg <- testData$negative/(length(negativeWords)) # number of _ words in lexicon

# add new column for number of words in each review to indicate length of review
wordcount <- function(str) {
  sapply(gregexpr("\\b\\W+\\b", str, perl=TRUE), function(x) sum(x>0) ) + 1 
}
testData$num_words <- wordcount(testData$sentence)

# add feature: difference between positive and negative words
testData$difference <- testData$positive - testData$negative

# add feature: proportion of positive to negative words
testData$prop_pos_neg <- ifelse(testData$negative == 0, 
                                       100, 
                                       testData$positive / testData$negative)

# add feature: proportion of positive or negative words to total words
testData$pos_totalwords <- testData$positive / testData$num_words
testData$neg_totalwords <- testData$negative / testData$num_words

# add feature: add the number of punctuation counted at beginning for each review
testData$periods[1:200] <- posP
testData$periods[201:400] <- negP
testData$exclamation_marks[1:200] <- posE
testData$exclamation_marks[201:400] <- negE
testData$q_marks[1:200] <- posQ
testData$q_marks[201:400] <- negQ

# make sure the values for the punctuation columns are numbers
testData$periods <- as.numeric(as.character(testData$periods))
testData$exclamation_marks <- as.numeric(as.character(testData$exclamation_marks))
testData$q_marks <- as.numeric(as.character(testData$q_marks))

# remove 'sentence' column from matrix as it will not be needed for RF; create new matrix
testMatrix <- testData
testMatrix$sentence <- NULL

# formula: RF, prediction
testResults <- predict(rfAnalysis, testMatrix)
testResults <- data.frame(testResults)
testResults$sentiment <- ifelse(testResults$positive > testResults$negative, 'positive', 'negative')
recall_accuracy(testMatrix$sentiment, testResults$sentiment)
# after adding new features, 68.25% accuracy with random forests
# 72.75% accuracy with decision tree

# 'CONCLUSIONS'
# Although decision tree had worse accuracy as compared to random forests when predictions were tested against
# original training data, it fared better when tested against test data, suggesting that the RF was overfitting

# let's trying automatic feature selection instead of using all the predictors

########
'AUTOMATIC FEATURE SELECTION'
########

# change column order so that 'sentiment' is first column
sentimentMatrix <- sentimentMatrix[, c(3,1,2,4:13)]

# try automating feature selection
library(caret)
ctrl <- rfeControl(method = "repeatedcv",
                   repeats = 5,
                   verbose = TRUE,
                   functions = rfFuncs)
set.seed(9)
featureSelection <- rfe(x = sentimentMatrix[,2:13],
                  y = sentimentMatrix$sentiment,
                  sizes = c(1:13),
                  metric = "Accuracy",
                  rfeControl = ctrl)
print(featureSelection)
# automated feature chose 3 variables: prop_pos_neg, difference, neg_totalwords, prop_pos, positive
# build new RF model using these 5 features
set.seed(10)
rfFeature <- rpart(sentiment ~ prop_pos_neg + difference + neg_totalwords + prop_pos + positive,
                           data = sentimentMatrix)

# TEST model
testResults2 <- predict(rfFeature, testData)
# results return a dataframe of percentages for chance of 'positive' or 'negative' sentiment
# turn percentages into factors for comparison
testResults2 <- data.frame(testResults2)
testResults2$sentiment <- ifelse(testResults2$positive > testResults2$negative, 'positive', 'negative')
recall_accuracy(testData$sentiment, testResults2$sentiment)

# accuracy 0.7275

# plot decision tree
fancyRpartPlot(rfFeature)
prp(rfFeature)

# with 5 automatically selected features:
# 68% accuracy with random forests
# 73% accuracy with decision tree

########
'Using Syuzhet package'
########

library(syuzhet)

# Import text files
setwd(paste(currentwd, "/postrain", sep=""))
pos <- list.files(path = setwd(paste(currentwd, "/postrain", sep="")), pattern = "*.txt")
positive.reviews <- lapply(pos, read_file)

setwd(paste(currentwd, "/negtrain", sep=""))
neg <- list.files(path = setwd(paste(currentwd, "/negtrain", sep="")), pattern = "*.txt")
negative.reviews <- lapply(neg, read_file)

# Below: test differences between different word lists (Bing, NRC, AFINN)

# sentiment analysis for all positive reviews
positive.reviews <- as.character(positive.reviews)
posSentiment <- get_sentiment(positive.reviews, method = "bing") # using "Bing" lexicon
posSentiment <- get_sentiment(positive.reviews, method = "afinn")
posSentiment <- get_sentiment(positive.reviews, method = "nrc")
length(which(posSentiment > 0))
# Percentage correct by lexicon:
# BING: 59.8%
# AFINN: 79.0%
# NRC: 90.3%

# sentiment analysis for all negative reviews
negative.reviews <- as.character(negative.reviews)
negSentiment <- get_sentiment(negative.reviews, method = "bing")
negSentiment <- get_sentiment(negative.reviews, method = "afinn")
negSentiment <- get_sentiment(negative.reviews, method = "nrc")
length(which(negSentiment < 0))
# Percentage correct by lexicon:
# BING: 77.1%
# AFINN: 50.1%
# NRC: 27.1%

################
'method without using word lists; using SVM and RF'
'ref: http://chengjun.github.io/en/2014/04/sentiment-analysis-with-machine-learning-in-R/'
################

library(RTextTools)
library(e1071)
library(readr)

setwd(paste(currentwd, "/postrain", sep=""))
pos <- list.files(path = setwd(paste(currentwd, "/postrain", sep="")), pattern = "*.txt")
positive.reviews <- lapply(pos, read_file)
positive.reviews <- lapply(positive.reviews, gsub, pattern = '[[:punct:],\n]', replacement = '')
positive.reviews <- lapply(positive.reviews, gsub, pattern = '\\d+', replacement = '')

posReviews <- cbind(positive.reviews, 'positive')

setwd(paste(currentwd, "/negtrain", sep=""))
neg <- list.files(path = setwd(paste(currentwd, "/negtrain", sep="")), pattern = "*.txt")
negative.reviews <- lapply(neg, read_file)
negative.reviews <- lapply(negative.reviews, gsub, pattern = '[[:punct:],\n]', replacement = '')
negative.reviews <- lapply(negative.reviews, gsub, pattern = '\\d+', replacement = '')

negReviews <- cbind(negative.reviews, 'negative')

allReviews <- rbind(posReviews, negReviews)
allReviews <- as.vector(allReviews)
colnames(allReviews) <- c('review', 'sentiment')

reviewsMatrix <- create_matrix(allReviews[,1], 
                               language="english",
                               removeStopwords = FALSE,
                               stemWords=FALSE,
toLower = TRUE) # turn words into lowercase

allReviews[,2] <- as.factor(as.character(allReviews[,2]))
# CLASSIFICATION: '2' is positive, '1' is negative

# create model to test first on training data
reviewsContainer <- create_container(reviewsMatrix, 
                              as.numeric(allReviews[,2]),
                              trainSize=1:2000, 
                              virgin=FALSE)

sentimentModels <- train_models(reviewsContainer, 
                       algorithms = c("SVM", "RF"),
                       cost = 5,
                       ntree = 20)

results1 <- classify_models(reviewsContainer, sentimentModels)

# test on training data
# SVM
recall_accuracy(as.numeric(allReviews[,2]), results1[,'SVM_LABEL']) # 81.8% accuracy
# RF
recall_accuracy(as.numeric(allReviews[,2]), results1[,'FORESTS_LABEL']) # 100% accuracy?!

# test data: http://ai.stanford.edu/~amaas/data/sentiment/

# build data frame for test data (using approx 400 text files, 50% positive)

# import data files
setwd(paste(currentwd, "/postest", sep=""))
pTest <- list.files(path = setwd(paste(currentwd, "/postest", sep="")), pattern = "*.txt")
positiveTest <- lapply(pTest, read_file)
positiveTest <- lapply(positiveTest, gsub, pattern = '[[:punct:],\n]', replacement = '')
positiveTest <- lapply(positiveTest, gsub, pattern = '\\d+', replacement = '')

setwd(paste(currentwd, "/negtest", sep=""))
nTest <- list.files(path = setwd(paste(currentwd, "/negtest", sep="")), pattern = "*.txt")
negativeTest <- lapply(nTest, read_file)
negativeTest <- lapply(negativeTest, gsub, pattern = '[[:punct:],\n]', replacement = '')
negativeTest <- lapply(negativeTest, gsub, pattern = '\\d+', replacement = '')

# consolidate +/- reviews into one large data frame (test data)
positiveTest <- cbind(positiveTest, 'positive')
negativeTest <- cbind(negativeTest, 'negative')
testData <- rbind(positiveTest, negativeTest)

# add test data to 'allReviews' for one data frame
testData[,2] <- as.factor(as.character(testData[,2]))
allReviews_test <- rbind(allReviews, testData)

# create model using new data frame
testMatrix <- create_matrix(allReviews_test[,1], 
                               language="english",
                               removeStopwords = FALSE,
                               stemWords=FALSE,
                               toLower = TRUE)

# identify data
testContainer <- create_container(testMatrix, 
                                     as.numeric(allReviews_test[,2]),
                                     trainSize=1:2000,
                                     testSize = 2001:2400, # test data
                                     virgin=FALSE)

# train model
testModels <- train_models(testContainer, 
                           algorithms = c("SVM", "RF"),
                           cost = 5,
                           ntree = 20)

# test model, see accuracy
results2 <- classify_models(testContainer, testModels)
# SVM
recall_accuracy(as.numeric(allReviews_test[2001:2400,2]), results2[,"SVM_LABEL"]) # 54% accuracy
# RF
recall_accuracy(as.numeric(allReviews_test[2001:2400,2]), results2[,"FORESTS_LABEL"]) # 69% accuracy

# Conclusions: I took my best performing model and tested it against the 400 new data files. 
# Results are not as high as I had anticipated (I was thinking maybe around 80% accuracy).
# This could be due to the fact that the test data text files were considerably shorter than the training data
# text files. 
