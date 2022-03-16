# MovieLens Project
# Author: David Brandtner
# Date: 21th of February 2022


# Loading required packages
requiredPackages <- c("tidyverse", "caret", "data.table", "patchwork", "knitr")
lapply(requiredPackages, library, character.only = TRUE)

###############################################################################
# Create edx set, validation set - code supplied by course assignation
###############################################################################

# Note: this process could take a couple of minutes

# if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
# if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
# if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
# 
# library(tidyverse)
# library(caret)
# library(data.table)

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")

# if using R 3.6 or earlier:
#movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
#                                           title = as.character(title),
#                                           genres = as.character(genres))
# if using R 4.0 or later:
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))


movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(1)`
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

###############################################################################
# Data Exploration and preprocessing
###############################################################################

# rating distribution
summary(edx$rating)

#Structur of edx dataset
str(edx)

#Preprocessing edx and validation set: creation of MovieYr column
edx_pr <- edx %>% mutate(movieYr = as.numeric(str_extract(title, "\\([0-9]{4}\\)") %>% 
                                                gsub("\\D","",.)))

validation_pr <- validation %>% mutate(movieYr = as.numeric(str_extract(title, "\\([0-9]{4}\\)") %>% 
                                                              gsub("\\D","",.)))

#Processing hypothetical predictors for plotting:
#movieId
edx_movieId <- edx_pr %>% group_by(movieId) %>% add_count(movieId) %>% mutate(m_rating=mean(rating)) %>%
  select(movieId, title, n, m_rating) %>% unique()

plot_movieId <- edx_movieId %>% ggplot(aes(m_rating, n)) + geom_point(size = 2/5) +
  scale_y_log10() + labs(x= "mean rating", y="times movie has been reviewed (log10)", title="Insight on movieId") +
  geom_smooth(orientation = "y", method = "gam")

#usersId
edx_userId <- edx_pr %>% group_by(userId) %>% add_count(userId) %>% mutate(m_rating=mean(rating)) %>%
  select(userId, n, m_rating) %>% unique()

plot_userId <- edx_userId %>% ggplot(aes(m_rating, n)) + geom_point(size = 2/5) +
  scale_y_log10() + labs(x= "mean rating", y="times user has reviewed (log10)", title="Insight on userId") +
  geom_smooth(orientation = "y", method = "gam")

#genres
edx_genres <- edx_pr %>% group_by(genres) %>% add_count(genres) %>% mutate(m_rating=mean(rating)) %>%
  select(genres, n, m_rating) %>% unique()

plot_genres <- edx_genres %>% ggplot(aes(m_rating, n)) + geom_point(size = 2/5) +
  scale_y_log10() + labs(x= "mean rating", y="times genre has movie reviewed (log10)", title="Insight on genres") +
  geom_smooth(orientation = "y", method = "gam")

#movie_Yr
edx_movieYr <- edx_pr %>% group_by(movieYr) %>% add_count(movieYr) %>% mutate(m_rating=mean(rating)) %>%
  select(movieYr, n, m_rating) %>% unique()

plot_movieYr <- edx_movieYr %>% ggplot(aes(m_rating, n)) + geom_point(size = 2/5) +
  scale_y_log10() + labs(x= "mean rating", y="times year has movie reviewed (log10)", title="Insight on movieYr") +
  geom_smooth(orientation = "y", method = "gam")

#Plotting using patchwork library
plot_movieId + plot_userId + plot_genres + plot_movieYr

rm(edx_movieId, edx_userId, edx_genres, edx_movieYr)

###############################################################################
# Training/Test dataset Splitting
###############################################################################

######## Splitting edx_pr in edx_pr.train_set and edx_pr.test_set
# edx_pr.test_set will be 10% of edx_pr data

set.seed(1, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(1)`

test_index <- createDataPartition(y = edx_pr$rating, times = 1, p = 0.1, list = FALSE)
edx_pr.train_set <- edx_pr[-test_index,]
temp <- edx_pr[test_index,]

# Make sure userId and movieId in edx_pr.test_set set are also in edx_pr.train_set set

edx_pr.test_set <- temp %>% 
  semi_join(edx_pr.train_set, by = "movieId") %>%
  semi_join(edx_pr.train_set, by = "userId")

# Add rows removed from edx_pr.test_set set back into edx_pr.train_set set

removed <- anti_join(temp, edx_pr.test_set)
edx_pr.train_set <- rbind(edx_pr.train_set, removed)

rm(test_index, temp, removed)

###############################################################################
# Modelling
###############################################################################



### Linear model adopted is composed by different parts. Here it is implemented
### functions for calculus of each part in order to have a cleaner code when
### it is applied on more than one dataset, like train, test and validation set.
### Adding function term "l" for further Regularization. Its default value is l=0,
### thus in this condition has no effect on calculus.

# rating_average: mu ------------------------------------------------

rating_avgs_f <- function(dataset){
  mu <- mean(dataset$rating)
  return(mu)
}

# Movie Effect: b_1 ----------------------------------------------

movie_avgs_f <- function(dataset, .rating_avgs, .l=0){
  b_1 <- dataset %>% group_by(movieId) %>% 
    summarize(b_1 = sum(rating - .rating_avgs)/(n()+.l))
  return(b_1)
}

# User Effect: b_2 --------------------------------------

user_avgs_f <- function(dataset, ..rating_avgs, ..movie_avgs, ..l=0){
  b_2 <- dataset %>% left_join(..movie_avgs, by = "movieId") %>%
    group_by(userId) %>% 
    summarize(b_2 = sum(rating - ..rating_avgs - b_1)/(n()+..l))
  return(b_2)
}

# Genres Effect: b_3 ------------------------------

genres_avgs_f <- function(dataset, ...rating_avgs, ...movie_avgs, ...user_avgs, ...l=0){
  b_3 <- dataset %>% left_join(...movie_avgs, by = "movieId") %>%
    left_join(...user_avgs, by = "userId") %>%
    group_by(genres) %>% 
    summarize(b_3 = sum(rating - ...rating_avgs - b_1 - b_2)/(n()+...l))
  return(b_3)
}

# Year Effect: b_4 -----------------------

year_avgs_f <- function(dataset, ....rating_avgs, ....movie_avgs, ....user_avgs, ....genres_avgs, ....l=0){
  b_4 <- dataset %>% left_join(....movie_avgs, by = "movieId") %>%
    left_join(....user_avgs, by = "userId") %>%
    left_join(....genres_avgs, by = "genres") %>%
    group_by(movieYr) %>% 
    summarize(b_4 = sum(rating - ....rating_avgs - b_1 - b_2 - b_3)/(n()+....l))
  return(b_4)
}


### In order to study the improvement carried by each model part 
### additive prediction functions are written. Functions are chosen
### to better organize the code and eliminating repetitive parts

# Model0: just the average ------------------------------
model0_f <- function(dataset){
  mu <- mean(dataset$rating)
  return(mu)
}

# Model1: Movie Effect Model ------------------------------
model1_f <- function(dataset, rating_avgs, movie_avgs){
  model_pred <- dataset %>% left_join(movie_avgs, by = "movieId") %>%
    mutate(pred = rating_avgs + b_1) %>%
    pull(pred)
  return(model_pred)
}

# Model2: Movie + User Effect Model ------------------------------
model2_f <- function(dataset, rating_avgs, movie_avgs, user_avgs){
  model_pred <- dataset %>% left_join(movie_avgs, by = "movieId") %>%
    left_join(user_avgs, by = "userId") %>%
    mutate(pred = rating_avgs + b_1 + b_2) %>%
    pull(pred)
  return(model_pred)
}

# Model3: Movie + User + Genres Effect Model ------------------------------
model3_f <- function(dataset, rating_avgs, movie_avgs, user_avgs, genres_avgs){
  model_pred <- dataset %>% left_join(movie_avgs, by = "movieId") %>%
    left_join(user_avgs, by = "userId") %>%
    left_join(genres_avgs, by = "genres") %>%
    mutate(pred = rating_avgs + b_1 + b_2 + b_3) %>%
    pull(pred)
  return(model_pred)
}

# Model4: Movie + User + Genres + Year Effect Model ------------------------------
model4_f <- function(dataset, rating_avgs, movie_avgs, user_avgs, genres_avgs, year_avgs){
  model_pred <- dataset %>% left_join(movie_avgs, by = "movieId") %>%
    left_join(user_avgs, by = "userId") %>%
    left_join(genres_avgs, by = "genres") %>%
    left_join(year_avgs, by = "movieYr") %>%
    mutate(pred = rating_avgs + b_1 + b_2 + b_3 + b_4) %>%
    pull(pred)
  return(model_pred)
}

# Creating the RMSE function
RMSE <- function(true_ratings, predicted_ratings) {
  sqrt(mean((true_ratings - predicted_ratings)^2))
}

####### TRAINING model on edx_pr.train_set
##########################################################
rating_avgs <- rating_avgs_f(edx_pr.train_set)

movie_avgs <- movie_avgs_f(edx_pr.train_set, rating_avgs)

user_avgs <- user_avgs_f(edx_pr.train_set, rating_avgs, movie_avgs)

genres_avgs <- genres_avgs_f(edx_pr.train_set, rating_avgs, movie_avgs, user_avgs)

year_avgs <- year_avgs_f(edx_pr.train_set, rating_avgs, movie_avgs, user_avgs, 
                         genres_avgs)


### TESTING models on edx_pr.test_set
###########################################################
# Model0:
mod0_pred <- model0_f(edx_pr.test_set)
# Model1:
mod1_pred <- model1_f(edx_pr.test_set, rating_avgs, movie_avgs)
#Model2:
mod2_pred <- model2_f(edx_pr.test_set, rating_avgs, movie_avgs, user_avgs)
#Model3:
mod3_pred <- model3_f(edx_pr.test_set, rating_avgs, movie_avgs, user_avgs,
                      genres_avgs)
#Model4:
mod4_pred <- model4_f(edx_pr.test_set, rating_avgs, movie_avgs, user_avgs,
                      genres_avgs, year_avgs)
#RMSE calculation
mod0_rmse <- RMSE(edx_pr.test_set$rating, mod0_pred)  
mod1_rmse <- RMSE(edx_pr.test_set$rating, mod1_pred)
mod2_rmse <- RMSE(edx_pr.test_set$rating, mod2_pred)
mod3_rmse <- RMSE(edx_pr.test_set$rating, mod3_pred)
mod4_rmse <- RMSE(edx_pr.test_set$rating, mod4_pred)

models_names <- c("Just the average", "Movie Effect Model", "Movie + User Effect Model",
                "Movie + User + Genres Effect Model", 
                "Movie + User + Genres + Year Effect Model")

models_id <- c("model0", "model1", "model2", "model3", "model4")

models_rmse <- c(mod0_rmse, mod1_rmse, mod2_rmse, mod3_rmse, mod4_rmse)

dataset_names <- c("edx_pr.test_set", "edx_pr.test_set", "edx_pr.test_set",
                   "edx_pr.test_set", "edx_pr.test_set")

rmse_results <- tibble(Model_Names = models_names, Model_Id = models_id, 
                       Dataset = dataset_names, RMSE = round(models_rmse, digit = 5))

#print Model/RMSE results
kable(rmse_results)


### Adding Effects Regularization to the more accurate Model, which is
### Model4: Movie + User + Genres + Year Effect Mode
###############################################################################

#Creating a tuning grid for lambdas "l" term in models functions:
lambdas <- seq(0, 6, 0.25)

### A new function for Model4 prediction is written integrating model  
### training and prediction on test set with a lambdas tuning grid for loop
### (penalty term "l"). Tuning grid results are saved in a vector.
### At the end RMSE calculation is integrated in the function too.

model4_reg_f <- function(train_dataset, test_dataset, lambdas){
  
  results <- c()
  
  for(z in lambdas){
    
    rating_avgs_r <- rating_avgs_f(train_dataset)
    
    movie_avgs_r <- movie_avgs_f(train_dataset, rating_avgs_r, .l=z)
    
    user_avgs_r <- user_avgs_f(train_dataset, rating_avgs_r, movie_avgs_r, ..l=z)
    
    genres_avgs_r <- genres_avgs_f(train_dataset, rating_avgs_r, movie_avgs_r,
                                   user_avgs_r, ...l=z)
    
    year_avgs_r <- year_avgs_f(train_dataset, rating_avgs_r, movie_avgs_r,
                               user_avgs_r, genres_avgs_r, ....l=z)
    
    mod4_pred_r <- model4_f(test_dataset, rating_avgs_r, movie_avgs_r, user_avgs_r,
                            genres_avgs_r, year_avgs_r)
    
    results <- append(results, RMSE(test_dataset$rating, mod4_pred_r))
    
  }
  return (results)
}

#Model4 + Regularization
#TRAINING on edx_pr.train_set + TESTING on edx_pr.test_set + lambda tuning grid.
#Results are saved into a vector.

mod4_rmse_results_reg <- model4_reg_f(edx_pr.train_set, edx_pr.test_set,lambdas)

#Selecting best lambdas value based on lower RMSE value
tuned_lambda <- lambdas[(which.min(mod4_rmse_results_reg))]

#Updating RMSE calculation
models_names <- append(models_names, "Movie + User + Genres + Year Effect Model + Reg")

models_id <- append(models_id, "model4_reg")

models_rmse <- append(models_rmse, min(mod4_rmse_results_reg))

dataset_names <- append(dataset_names, "edx_pr.test_set")

rmse_results <- tibble(Model_Names = models_names, Model_Id = models_id, 
                       lambda = c("NA", "NA", "NA", "NA", "NA", tuned_lambda),
                       Dataset = dataset_names, RMSE = round(models_rmse, digit = 5))

#print Model/RMSE results
kable(rmse_results)

###############################################################################
#Results
###############################################################################

# Final RMSE calculation on validation_pr dataset with best model

mod4_rmse_validation_pr_reg <- model4_reg_f(edx_pr.train_set, validation_pr, tuned_lambda)

#Updating RMSE calculation
models_names <- append(models_names, "Movie + User + Genres + Year Effect Model + Reg")

models_id <- append(models_id, "model4_reg")

models_rmse <- append(models_rmse, mod4_rmse_validation_pr_reg)

dataset_names <- append(dataset_names, "validation_pr")

rmse_results <- tibble(Model_Names = models_names, Model_Id = models_id, 
                       lambda = c("NA", "NA", "NA", "NA", "NA", tuned_lambda, tuned_lambda),
                       Dataset = dataset_names, RMSE = round(models_rmse, digit = 5))

#print Model/RMSE results
kable(rmse_results)
