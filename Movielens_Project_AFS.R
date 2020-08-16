

###Packages Download
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(data.table)
library(dplyr)


####Dataset downloading and partition for ML: edx set, validation set
# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")

movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding")
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# UserId and movieId must be in both validation set and in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)
rm(dl, ratings, movies, test_index, temp, movielens, removed)


###Exploring Data Set [Training Set]
str(cars)
head(edx)
summary(edx)

## Modifications of  both  data set: Year of the movie and unique genre:
#Year: We substract the last 4 strings of the movie title, omiting the "()", and then transform it into
#numeric form:
edx <- edx %>% mutate(year = as.numeric(str_sub(title,-5,-2)))
validation <- validation %>% mutate(year = as.numeric(str_sub(title,-5,-2)))
#Genre: We generete new rows for each genre identify per movie.
edx<- edx  %>% separate_rows(genres, sep = "\\|")

head(edx)
summary(edx)


###Some insigths of the data set:
##Unique Movies and Unique Users
edx %>% summarize(n_users = n_distinct(userId), n_movies = n_distinct(movieId))

##Movies per gendre:
drama <- edx %>% filter(str_detect(genres,"Drama"))
comedy <- edx %>% filter(str_detect(genres,"Comedy"))
thriller <- edx %>% filter(str_detect(genres,"Thriller"))
romance <- edx %>% filter(str_detect(genres,"Romance"))
Num_drama<-nrow(drama)
Num_comedy<-nrow(comedy)
Num_thriller<-nrow(thriller)
Num_romance<-nrow(romance)


##Distribution:
#1. Distribution of the Movie Ratings 
edx %>% 
  ggplot(aes(rating)) + 
  geom_histogram(binwidth=0.5, color="black", fill="grey") + 
  ggtitle("Rating Distribution")

edx %>% group_by(movieId) %>% summarize(n = n()) %>%
  ggplot(aes(n)) + 
  geom_histogram(fill = "grey", color = "black", bins = 10) +
  scale_x_log10() +
  ggtitle("Distribution Ratings by Movies")

#2. Distribution of Users
edx %>% group_by(userId) %>% summarize(n = n()) %>%
  ggplot(aes(n)) + 
  geom_histogram(fill = "grey", color = "black", bins = 10) +
  scale_x_log10() + 
  ggtitle("Distribution Users Ratings")


#3. Distribution of Rating per Year
edx%>%group_by(year) %>% summarize(n = n()) %>%
  ggplot(aes(x = year, y = n)) +
  geom_line(color="grey")

#release year vs rating
edx %>% group_by(year) %>%
  summarize(mean_rating= mean(rating)) %>%
  ggplot(aes(year, mean_rating )) +
  geom_point() +
  geom_smooth() +
  ggtitle("Year vs. Rating")

#4. Distribution of Rating by genre
#By rating
edx %>%group_by(genres) %>% summarize(mean_rating_genre = mean(rating))%>%
  ggplot(aes(mean_rating_genre,genres)) +
  geom_point() +
  ggtitle("Mean rating Genre ")

#By number of users
edx %>% 
  ggplot(aes(count(userId),genres)) +
  geom_point() +
  ggtitle("Mean rating Genre")


####RMSE:FUNCTION

RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}

#1.Baseline (simplest one)
mu <- mean(edx$rating)
naive_RMSE <- RMSE(validation$rating, mu)
naive_RMSE

#2.Adding Movie effect model (Taking into account b_i(movie effect))

  #The movie effect:The mean of substracting the mean to the rating (b_i)
  movie_effect <- edx %>% 
    group_by(movieId) %>% 
    summarize(b_i = mean(rating - mu))
  
  #Predictions: We compute the predicted movie effect in the validation dataset
  predicted_movie_effect <- validation %>% 
    left_join(movie_effect, by='movieId') %>%
    mutate(prediction = mu + b_i) 
  movie_RMSE <- RMSE(validation$rating,predicted_movie_effect$prediction)


#3.Adding User effect model to previous model (Taking into account b_i(movie effect))
  #First, we add the movie effect, then group by userId, and calculate b_u: The mean of substracting the mean and b_i to the rating. 
  user_effect <- edx %>% 
    left_join(movie_effect, by='movieId') %>%
    group_by(userId) %>%
    summarize(b_u = mean(rating - mu - b_i))
  
  #Predictions: We compute the predicted user+ movie effect in the validation dataset
  predicted_movie_user_effect  <- validation %>% 
    left_join(movie_effect, by='movieId') %>%
    left_join(user_effect, by='userId') %>%
    mutate(prediction = mu + b_i + b_u) 
  
  movie_user_RMSE <- RMSE(validation$rating,predicted_movie_user_effect$prediction)

#4.Adding Year effect model to previous model (Taking into account b_i(movie effect))
  #First, we add the movie and user effect, then group by year, and calculate b_y 
  year_effect <- edx %>% 
    left_join(movie_effect, by='movieId') %>%
    left_join(user_effect, by='userId') %>%
    group_by(year) %>%
    summarize(b_y = mean(rating - mu - b_i - b_u))
  #Predictions: We compute the predicted user+movie+year effect in the validation dataset
  predicted_movie_user_year_effect  <- validation %>% 
    left_join(movie_effect, by='movieId') %>%
    left_join(user_effect, by='userId') %>%
    left_join(year_effect, by='year') %>%
    mutate(prediction = mu + b_i + b_u + b_y) 
  
  movie_user_year_RMSE <- RMSE(validation$rating,predicted_movie_user_year_effect$prediction)


##Since the data is extremely infuenced by noisy estimate as we notice during the data exploration:
#Ex: Users that made few reviews, Movies with few reviews. We need to remove  the effect
#of these noise effect as possible in order to improve our RMSE. Therefore we must chose a lambda that fits better
#our model.

lambdas <- seq(0,10,0.2)
RMSEs <- sapply(lambdas, function(lambda){
  mu <- mean(edx$rating)
  
  b_i <- edx %>%
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n() + lambda))
  
  b_u <- edx%>%
    left_join(b_i, by='movieId') %>% 
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n() +lambda))
  
  b_y <- edx%>%
    left_join(b_i, by='movieId') %>% 
    left_join(b_u, by='userId') %>% 
    group_by(year) %>%
    summarize(b_y = sum(rating - b_i - b_u - mu)/(n() +lambda))
  
  predicted_ratings <- edx %>%
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    left_join(b_y, by = "year") %>%
    mutate(prediction = mu + b_i +b_u + b_y) %>% .$prediction
  
  return(RMSE(predicted_ratings, edx$rating))
})

qplot(lambdas, RMSEs)


######----MODELS RESULTS-----########                                                         
rmse_results <- data_frame(Model=c("Naive","Movie Effect","Movie+User Effect","Movie+User+Year Effect","Regularized Movie+User+Year Effect"), 
                           RMSE = c(naive_RMSE,movie_RMSE,movie_user_RMSE,movie_user_year_RMSE,min(RMSEs)))
rmse_results%>%knitr::kable(format="rst")