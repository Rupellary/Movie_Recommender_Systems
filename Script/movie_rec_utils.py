import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from lifelines.utils import concordance_index
import difflib

def get_rating_count(user_or_movie, ratings_df):
    """Description: Counts the number of ratings made by each user or of each movie
    INPUTS: user_or_movie = index associated with either users or movies to get counts of ratings per user or movie respectively
            ratings_df = the data frame with ratings users have made of movies
    OUTPUTS: rating_count_df = data frame with counts of ratings per user or movie depending on input"""
    ratings_df['rating_count'] = 1
    rating_count_df = ratings_df.groupby(user_or_movie).agg({'rating_count':sum})
    return rating_count_df


def count_shared_users(movie_list, pivoted_df):
    """Description: Counts how many users have reviewed all the movies in the input list
    INPUTS: movies_list = list of movies to check for common reviewers
            pivoted_df = data frame with users as rows and movies as columns
    OUTPUTS: count of users that have reviewed all of the movies in the list"""
    movies = pivoted_df[movie_list]
    movies.loc[movies.notna().all(axis=1), 'shared'] = 1
    return movies['shared'].sum()


def find_by_name(title, movies_df):
    closest_match = difflib.get_close_matches(title, movies_df['title'], n=1, cutoff=0.6)
    if closest_match:
        return closest_match[0]
    else:
        return None


def find_similar(movie_id, ratings_df, movies_df, pivoted_df, n, shared_thresh=3, total_thresh=10):
    """Description: Gets a list of similar movies to an input movie id using item-based filtering
    INPUTS: movie_id = id of movie to find movies similar to
            ratings_df =  data frame with ratings, user ids, and movie ids as columns
            movies_df = data frame with movies with their ids and titles
            pivoted_df = data frame with users as rows and movies as columns
            n = number of similar movies to return
            shared_thresh = in order for a movie to be recommended it must have been reviewed by at least this number of users who have also reviewed the input movie
            total_thresh = minimum number of reviews a movie must have before it can be recommended
    OUTPUTS: topn_df = data frame with movie id, the correlation score, the number of ratings the movie has, the number of shared users between movies, the title and the genres for the top n movies according to the n input"""
    #id = find_by_name(title)
    ratings = pivoted_df[movie_id]
    corrs = pivoted_df.corrwith(ratings)
    corrs_df = pd.DataFrame(corrs, columns=['PearsonR']).dropna(inplace=True)
    rating_count_df = get_rating_count(movie_id, ratings_df)
    corrs_and_count = corrs_df.merge(rating_count_df, how='left', on='movieId')
    corrs_and_count['shared_count'] = corrs_and_count.apply(lambda row: count_shared_users(row['movieId'],id), axis=1)
    corrs_and_count = corrs_and_count[corrs_and_count['shared_count']>=shared_thresh]
    corrs_and_count = corrs_and_count.set_index('movieId')
    corrs_and_count.drop(movie_id, inplace=True)
    topn_corr = corrs_and_count[corrs_and_count['rating_count']>=total_thresh].sort_values('PearsonR', ascending=False).head(n)
    topn_df = topn_corr.merge(movies_df, how='left', on='movieId')
    return topn_df


def create_train(ratings_df, train_split):
    """Description: Creates a data frame with users as rows and movies as columns filled only with training data
    INPUTS: ratings_df = data frame with ratings as rows, must be indexed with 'userId', 'movieId', and 'rating'
            train_split = training portion of the same data frame
    OUTPUT: useritem_train = data frame with training data reviews"""
    # Ensuring no users or movies are lost from original dataset during the split
    users = sorted(ratings_df['userId'].unique())
    items = sorted(ratings_df['movieId'].unique())
    # Creating empty data frame with movies as columns and users as rows
    useritem_train = pd.DataFrame(0, index=users, columns=items)
    # Filling data frame with data in the train dataset
    for index, row in train_split.iterrows():
        useritem_train.loc[row['userId'], row['movieId']] = row['rating']
    return useritem_train


def estimate_rating(train_df, userID, movieID, similarity_df):
    """Description: estimates a missing rating a given user might give for a given movie
    INPUTS: train_df = data frame of training data with users as rows and movies as columns
            userID = user to be recommended ie. predicted for
            movieID = movie to be estimated
            similarity_df = data frame with similarity scores between users
    OUTPUTS: rating = estimated rating the input user would give to the input movie"""
    # Creating weights based on user similarity
    sim_scores = similarity_df[userID]
    sim_scores = sim_scores.drop(userID) # drop self-similarity
    weighted_scores = sim_scores / sim_scores.sum()
    # Computing weighted average of other users' ratings
    movie_ratings = train_df.loc[:, movieID]
    rating = (weighted_scores * movie_ratings).sum()
    return rating


def score_est(ratings, estimations):
    """Description: computes several performance metrics for evaluating a set of predictions
    INPUTS: ratings = the actual ratings in the test set
            estimations = the estimations used by the recommender system
    OUTPUTS: scoring_df = data frame with three computed performance metrics"""
    mae = mean_absolute_error(ratings, estimations)
    c_i = concordance_index(ratings, estimations)
    r2 = r2_score(ratings, estimations)
    scoring_df = pd.DataFrame({'Mean Absolute Error':[mae],
                               'Concordance Index':[c_i],
                               'R Squared':[r2]})
    return scoring_df