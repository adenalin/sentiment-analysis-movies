# Intro: I will be using my 'simple' algorithm model (# positive vs # negative words) for sentiment analysis
# as previously, that had the best overall performance with an accuracy of approximately 70%

# PRESENTATION LINK: http://rpubs.com/adena/sentiment-web

# set working directory to wherever file 'Sentiment Analysis' is located
currentwd <- setwd("/Users/Adena/Desktop/Sentiment Analysis/")

# load necessary packages
library(rvest)
library(readr)
library(plyr)
library(stringr)
library(RTextTools)

# import +/- word lists
positiveWords <- scan("lexicon/positive-words.txt",
                      what = 'character')
positiveWords[2007] <- c('funny')
negativeWords <- scan("lexicon/negative-words.txt",
                      what = 'character')
negativeWords[4784] <- c('tough')

# enter website url (find professor's rating page) to scrape
# -----> PROF 1 <-----
uwaterloo <- read_html("http://www.ratemyprofessors.com/ShowRatings.jsp?tid=1793811")
webData <- uwaterloo %>%
  html_nodes(".commentsParagraph") %>%
  html_text() %>%
  as.character()
print(webData)
# this should give a list of reviews from the first page of the website

# process reviews (remove punctuation and line breaks) to run through model
reviews <- lapply(webData, gsub, pattern = '[[:punct:],\n,\r]', replacement = '')
# change all words to lowercase
reviews <- tolower(reviews)

# Function to break down reviews into words for sentiment analysis
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

# data frame of positive/negative sentences with scores
# this compares words in reviews to lexicon to count number of +/- words
results <- as.data.frame(sentimentFunction(reviews, positiveWords, negativeWords))
colnames(results) <- c('review', 'positive', 'negative')
# end up with data frame that lists the number of +/- words in each review

# remove blank reviews
results <- results[!(gsub("[[:space:]]", "", results$review) == ""), ]

# make sure +/- words are represented as numbers
results$positive <- as.numeric(as.character(results$positive))
results$negative <- as.numeric(as.character(results$negative))

# formula: if #positive words > #negative words, then 'positive'
# equal values = 'neutral'
for(i in 1:length(results$review)) {
  if (results$positive[i] > results$negative[i]) {
    results$predicted[i] <- "positive"
  } else if (results$positive[i] < results$negative[i]) {
    results$predicted[i] <- "negative"
  } else {
    results$predicted[i] <- "neutral"
  }
}

# get prof 'helpfulness' rating
helpfulness <- uwaterloo %>%
  html_nodes(".break:nth-child(1) .score") %>%
  html_text() %>%
  as.numeric()
print(helpfulness)

# get prof 'clarity' rating
clarity <- uwaterloo %>%
  html_nodes(".break:nth-child(2) .score") %>%
  html_text() %>%
  as.numeric()
print(clarity)

# put helpfulness and clarity rating into dataframe
ratings <- data.frame(helpfulness, clarity)

# find overall rating by averaging the 2 ratings
ratings$overall <- (helpfulness + clarity)/2

# set sentiment ranges based on overall ratings (same manner as RateYourProf)
ratings$sentiment[ratings$overall < 3] <- 'negative'
ratings$sentiment[ratings$overall >= 3 & ratings$overall <= 3.5] <- 'neutral'
ratings$sentiment[ratings$overall > 3.5] <- 'positive'

# import csv file of objective ratings based on reviews alone (not rated based on helpfulness/clarity)
objRatings <- read.csv("sentimentset_dbm.csv", header = FALSE)
# extract ratings for respective prof (csv file has 3 profs) 
# <----- this part needs to be adjusted based on prof ----->
objRatings <- data.frame(objRatings[3:19,])
colnames(objRatings) <- c('sentiment', 'review')

# TEST
# compare prediction (based on word sentiment) with actual numeric ratings
recall_accuracy(ratings$sentiment, results$predicted)
# compare predicted vs. objective ratings
recall_accuracy(objRatings$sentiment, results$predicted)
# compare numeric (website) ratings with objective ratings
recall_accuracy(ratings$sentiment, objRatings$sentiment)

# 76% accuracy with PROF #1 (Bergsieker 'http://www.ratemyprofessors.com/ShowRatings.jsp?tid=1793811')
# 80% accuracy with PROF #2 (Kroeker 'http://www.ratemyprofessors.com/ShowRatings.jsp?tid=870995')
# 82% accuracy with PROF #3 (Goldberg 'http://www.ratemyprofessors.com/ShowRatings.jsp?tid=1076354')

# this data frame was used for tweaking the model
# ie. adjusting range for 'neutral' reviews, or number of words for sentiment classification, 
# or adding certain common words not found in lexicon
compare <- data.frame(results$positive,
                      results$negative,
                      ratings$overall, 
                      ratings$sentiment, 
                      results$predicted)
