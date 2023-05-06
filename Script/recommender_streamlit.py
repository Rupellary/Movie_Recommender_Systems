import streamlit as st
from movie_rec_utils import *

# Commandline Prompt to Run App:
# streamlit run C:\Users\rockw\WBS\RecommenderSystems\Script\recommender_streamlit.py

# Read Data
movies_df = pd.read_csv('C:/Users/rockw/WBS/RecommenderSystems/Data/movies.csv')
ratings_df = pd.read_csv('C:/Users/rockw/WBS/RecommenderSystems/Data/ratings.csv')

# Header Text
st.title('Movie Recommender')
st.write("""
### Project description
Below is a simple recommender system that accesses a database of movies and user reviews to make recommendations
""")


# Item-Based Collaborative Filtering

st.write("""
### Item-Based Collaborative Filtering
Use a title name to find other movies that users reviewed similarly
""")

# User inputs a title
title = st.text_input(label='Write the title of a movie you enjoyed',
                      value='The Godfather')
# Get the title as it is in the database and confirm it with the user
if find_by_name(title, movies_df, thresh=0.6) == None:
    st.markdown('**:red[Im sorry, this title could not be found. Try adding the subtitle and/or using roman numerals (eg. "Star Wars 4" -> "Star Wars Episode IV - A New Hope")]**')
else:
    movie_id = int(movies_df.loc[movies_df['title'] == find_by_name(title, movies_df, thresh=0.6), 'movieId'])
    real_title = movies_df.loc[movies_df['movieId']==movie_id].reset_index()['title'][0]
    st.markdown(f'**Displaying titles similar to {real_title}**')

# User inputs number of titles to recommend
num = st.number_input(label='How many movies do you want recommended?', 
                      value=5, 
                      step=1)

# User inputs whether or not they would like more data than the title
more_data = st.selectbox('Would you like additional data on the correlations?', ['Yes, more data please', 'No, just the titles'])
if more_data == 'Yes, more data please':
    extra_data = True
else: extra_data = False    

# Search button actually runs the function and displays the outputs
clicked = st.button(label='Search') 

if clicked == True:
    st.write(item_based_rec(title, 
                            ratings_df=ratings_df,
                            movies_df=movies_df,
                            n=num,
                            shared_thresh=20,
                            total_thresh=15,
                            more_data=extra_data))
    clicked = False


# User-Based Collaborative Filtering
#st.write("""
### Item-Based Collaborative Filtering
#Get shown random titles and rate them, then see movies that users who gave similar reviews liked
#""")


# What if we presented a title from the database at random and gave the user 
# the option to rate or pass. Then these ratings would be stored under a new user 
# and a user-based collaborative filtering could be employed

# review = st.selectbox('Review', [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
