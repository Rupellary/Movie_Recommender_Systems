import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_absolute_error, r2_score
from lifelines.utils import concordance_index
import difflib
import numpy as np
import re



# ASSIST FUNCTIONS

def count_shared_users(movie_list, pivoted_df):
    """Description: Counts how many users have reviewed all the movies in the input list
    INPUTS: movies_list = list of movies to check for common reviewers
            pivoted_df = data frame with users as rows and movies as columns
    OUTPUTS: count of users that have reviewed all of the movies in the list
    """
    movies = pivoted_df[movie_list]
    movies.loc[movies.notna().all(axis=1), 'shared'] = 1
    return movies['shared'].sum()

def find_by_name(title, movies_df, thresh=0.6):
    """Description: matches imperfect title inputs with closest title in dataset
    INPUTS: title = string that is close to the stored title
            movies_df = dataframe with movie titles
            thresh = specifies how similar the word must be to match
    OUTPUTS: closest title in the dataset as it is written in the dataset, or None if none can be found
    """
    # Capitalize certain words
    low_case_words = ["the", "of", "and", "to"]
    words = title.split(' ')
    for i in range(len(words)):
        if words[i] not in low_case_words:
            words[i] = words[i].capitalize()
        else: continue
    title = ' '.join(words)
    # If the title starts with 'The', move it to the end of the string
    if re.match(r'^the', title, re.IGNORECASE):
        adjusted_title = title[3:] + ', The' # adjust input titles with "The" to the format used by the database
    else: adjusted_title = title
    # Use difflib library to find closest strings in the dataset
    closest_match = difflib.get_close_matches(adjusted_title, movies_df['title'], n=1, cutoff=thresh)
    if closest_match:
        return closest_match[0]
    else:
        return None

def get_rating_count(user_or_movie, ratings_df):
    """Description: Counts the number of ratings made by each user or of each movie
    INPUTS: user_or_movie = index associated with either users or movies to get counts of ratings per user or movie respectively
            ratings_df = the data frame with ratings users have made of movies
    OUTPUTS: rating_count_df = data frame with counts of ratings per user or movie depending on input
    """
    ratings_df['rating_count'] = 1
    rating_count_df = ratings_df.groupby(user_or_movie).agg({'rating_count':sum})
    return rating_count_df



# POPULARITY RANKERS

def get_pop_rev(ratings_df, movies_df, n, review_thresh):
    """Description: averages all ratings per movie and displays only those with more than an input number of ratings
    INPUTS: ratings_df = data frame with ratings as rows including movie ids, user ids, and rating values as columns
            movies_df = data frame with movie information including movie ids and titles
            n =  number of recommendations to return
            review_thresh = minimum number of reviews a movie must have to be considered
    OUTPUTS: list of n top movies
    """
    ratings_df['rating_count'] = 1
    mean_ratings = ratings_df.groupby('movieId').agg({'rating':np.mean, 'rating_count':sum})
    mean_ratings = mean_ratings.loc[mean_ratings['rating_count']>=review_thresh]
    top_movies = mean_ratings.sort_values('rating', ascending=False).head(n)
    top_movie_titles = top_movies.merge(movies_df, how='left', on='movieId')
    return top_movie_titles['title']

def get_pop_laplace(ratings_df, movies_df, n, rat=0.5, num_fake=2):
    """Description: adds a small number of fake negative reviews to all movies more strongly punishing those with fewer reviews so that the average becomes more reflective
    INPUTS: ratings_df = data frame with ratings as rows including movie ids, user ids, and rating values as columns
            movies_df = data frame with movie information including movie ids and titles
            n =  number of recommendations to return
            rat = value of negative rating to be added to all movies
            num_fake = number of negative reviews to add to each movie
    OUTPUTS: list of n top movies
    """
    new_ratings_df = ratings_df[['movieId', 'rating']]
    movies = movies_df['movieId'].unique()
    for mId in movies:
        for i in range(0,num_fake):
            new_rating = pd.DataFrame({
                'movieId':[mId],
                'rating':[rat]
            })
            new_ratings_df = pd.concat([new_ratings_df, new_rating])
    top_titles = get_pop_rev(ratings_df=new_ratings_df, movies_df=movies_df, n=n, review_thresh=0)
    return top_titles

def get_pop_cumulative(ratings_df, movies_df, n):
    """Description: sums all ratings to get a measure that takes into account quantity of ratings and score into popularity measure
    INPUTS: ratings_df = data frame with ratings as rows including movie ids, user ids, and rating values as columns
            movies_df = data frame with movie information including movie ids and titles
            n =  number of recommendations to return
    OUTPUTS: list of n top movies
    """
    ratings_cumu = ratings_df.groupby('movieId').agg({'rating':sum})
    top_movies = ratings_cumu.sort_values('rating', ascending=False).head(n)
    top_titles = top_movies.merge(movies_df, how='left', on='movieId')['title']
    return top_titles



# COLLABORATIVE FILTERS

def item_based_rec(title, ratings_df, movies_df, n, shared_thresh=3, total_thresh=10, more_data=False):
    """Description: Gets a list of similar movies to an input movie id using item-based filtering
    INPUTS: title = approximate title of movie to search for movies similar to
            ratings_df =  data frame with ratings, user ids, and movie ids as columns
            movies_df = data frame with movies with their ids and titles
            n = number of similar movies to return
            shared_thresh = in order for a movie to be recommended it must have been reviewed by at least this number of users who have also reviewed the input movie
            total_thresh = minimum number of reviews a movie must have before it can be recommended
            more_data = if False will return only titles, if True will return data frame with movie id, the correlation score, the number of ratings the movie has, the number of shared users between movies, the title and the genres
    OUTPUTS: topn_df = returns specified number of most similar movies, number of columns based on input for more_data argument
    """
    if find_by_name(title, movies_df, thresh=0.6) == None:
        error_string= 'Error: Title not recognized. Try adding subtitle (eg. "Star Wars" --> "Star Wars Episode IV - A New Hope")'
        print(error_string)
        #return error_string
    else:
        # pivot ratings data frame into one with movies as columns, users as rows, and ratings as cell values
        pivoted_df = pd.pivot_table(ratings_df, 
                                    values='rating', 
                                    index='userId', 
                                    columns='movieId')
        # get movie id using name
        movie_id = int(movies_df.loc[movies_df['title'] == find_by_name(title, movies_df, thresh=0.6), 'movieId'])
        # get dataset's title to print with
        real_title = movies_df.loc[movies_df['movieId']==movie_id].reset_index()['title'][0]
        print(f"Top {n} titles similar to {real_title}:")
        # get ratings of movie from all users
        ratings = pivoted_df[movie_id]
        # corrlate user ratings of input movie with user ratings of all other movies
        corrs = pivoted_df.corrwith(ratings)
        corrs_df = pd.DataFrame(corrs, columns=['PearsonR'])
        corrs_df = corrs_df.loc[corrs_df['PearsonR'].isna() == False]
        # get number of ratings for each movie
        rating_count_df = get_rating_count('movieId', ratings_df).reset_index()
        corrs_and_count = corrs_df.merge(rating_count_df, how='left', on='movieId')
        # filter for movies with number of ratings above input threshold
        corrs_and_count = corrs_and_count[corrs_and_count['rating_count']>=total_thresh]
        # get number of users for each movie that have reviewed both the given movie and the input movie
        corrs_and_count['shared_count'] = corrs_and_count.apply(lambda row: count_shared_users([row['movieId'],movie_id],pivoted_df=pivoted_df), axis=1)
        # filter for movies with more shared users than the input threshold
        corrs_and_count = corrs_and_count[corrs_and_count['shared_count']>=shared_thresh]
        corrs_and_count = corrs_and_count.set_index('movieId')
        # remove input movie from list
        corrs_and_count.drop(movie_id, inplace=True)
        # sort by degree of correlation
        topn_corr = corrs_and_count.sort_values('PearsonR', ascending=False).head(n)
        # get additional movie information including titles
        topn_df = topn_corr.merge(movies_df, how='left', on='movieId')
        if more_data == True:
            return topn_df.drop(columns='movieId')
        # if additional data is not wanted, return only title column
        else: return topn_df['title']

def estimate_rating(pivoted_df, userID, movieID, similarity_df):
    """Description: estimates a missing rating a given user might give for a given movie
    INPUTS: train_df = pivoted data frame with users as rows and movies as columns
            userID = user to be recommended ie. predicted for
            movieID = movie to be estimated
            similarity_df = data frame with similarity scores between users
    OUTPUTS: rating = estimated rating the input user would give to the input movie
    """
    # Creating weights based on user similarity
    sim_scores = similarity_df[userID]
    sim_scores = sim_scores.drop(userID) # drop self-similarity
    weighted_scores = sim_scores / sim_scores.sum()
    # Computing weighted average of other users' ratings
    movie_ratings = pivoted_df.loc[:, movieID]
    rating = (weighted_scores * movie_ratings).sum()
    return rating

def user_based_rec(user_id, ratings_df, movies_df, n, more_data = False):
    """Description: correlates similar users and more highly weights their ratings to calculate recommendations
    INPUTS: user_id = id of user to recommend to
            ratings_df = data frame with user ids, movie ids, and ratings, one row per rating
            movies_df = data frame with at least movie ids and titles
            n =  number of movies to recommend
            more_data = if False simply returns titles, if True will return a data frame with additional information including the scoring of each movie
    OUTPUTS: either data frame with more information or just a list of titles depending on the more_data input
    """
    pivoted_df = pd.pivot_table(ratings_df, 
                                values='rating', 
                                index='userId', 
                                columns='movieId').fillna(0)
    # compute user similarity
    cos_sim_df = pd.DataFrame(cosine_similarity(pivoted_df, pivoted_df), 
                              index=pivoted_df.index, 
                              columns=pivoted_df.index)
    # get list of movie ids
    movies = sorted(ratings_df['movieId'].unique())
    estimations_df = pd.DataFrame({'movieId':movies})
    # use custom estimation function
    estimations_df['estimated_rating'] = estimations_df.apply(lambda row: estimate_rating(pivoted_df, user_id, row['movieId'], cos_sim_df), axis=1)
    # get top n
    best_recs = estimations_df.sort_values('estimated_rating', ascending=False).head(n)
    # get additional movie information
    best_recs = best_recs.merge(movies_df, how='left', on='movieId')
    if more_data == True:
        return best_recs
    else: return best_recs['title']

def create_train(ratings_df, train_split):
    """Description: Creates a data frame with users as rows and movies as columns filled only with training data
    INPUTS: ratings_df = data frame with ratings as rows, must be indexed with 'userId', 'movieId', and 'rating'
            train_split = training portion of the same data frame
    OUTPUT: useritem_train = data frame with training data reviews
    """
    # Ensuring no users or movies are lost from original dataset during the split
    users = sorted(ratings_df['userId'].unique())
    items = sorted(ratings_df['movieId'].unique())
    # Creating empty data frame with movies as columns and users as rows
    useritem_train = pd.DataFrame(0, index=users, columns=items)
    # Filling data frame with data in the train dataset
    for index, row in train_split.iterrows():
        useritem_train.loc[row['userId'], row['movieId']] = row['rating']
    return useritem_train

def score_est(ratings, estimations):
    """Description: computes several performance metrics for evaluating a set of predictions
    INPUTS: ratings = the actual ratings in the test set
            estimations = the estimations used by the recommender system
    OUTPUTS: scoring_df = data frame with three computed performance metrics
    """
    mae = mean_absolute_error(ratings, estimations)
    c_i = concordance_index(ratings, estimations)
    r2 = r2_score(ratings, estimations)
    scoring_df = pd.DataFrame({'Mean Absolute Error':[mae],
                               'Concordance Index':[c_i],
                               'R Squared':[r2]})
    return scoring_df



# STREAMLIT

# This function would be for user-based collaborative filtering on streamlit, but this part of the project is unlikely to be finished
def get_random_title(movies_df, thresh):
    """Description: returns a random title in the database that has at least as many reviews as the specified threshold
    INPUTS: movies_df = a data frame with the movies to be randomly selected from
            thresh = the minimum number of reviews a movie must have to be considered
    OUTPUTS: random_title = the title of a movie"""
