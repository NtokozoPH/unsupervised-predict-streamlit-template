"""

    Content-based filtering for item recommendation.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within the root of this repository for guidance on how to use
    this script correctly.

    NB: You are required to extend this baseline algorithm to enable more
    efficient and accurate computation of recommendations.

    !! You must not change the name and signature (arguments) of the
    prediction function, `content_model` !!

    You must however change its contents (i.e. add your own content-based
    filtering algorithm), as well as altering/adding any other functions
    as part of your improvement.

    ---------------------------------------------------------------------

    Description: Provided within this file is a baseline content-based
    filtering algorithm for rating predictions on Movie data.

"""

# Script dependencies
import os
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

# Importing data
movies = pd.read_csv('resources/data/movies.csv', sep = ',',delimiter=',')
ratings = pd.read_csv('resources/data/ratings.csv')
imdb_data = pd.read_csv('../unsupervised_data/unsupervised_movie_data/imdb_data.csv') #'../unsupervised_data/unsupervised_movie_data
movies_df = imdb_data.merge(movies, left_on='movieId', right_on='movieId')
movies_df.drop(['runtime','budget','movieId'],axis=1,inplace=True)
#movies_df.fillna('')
#movies.dropna(inplace=True)

def data_preprocessing(subset_size):
    """Prepare data for use within Content filtering algorithm.

    Parameters
    ----------
    subset_size : int
        Number of movies to use within the algorithm.

    Returns
    -------
    Pandas Dataframe
        Subset of movies selected for content-based filtering.

    """
    # function for changing column data types to strings for string handling
    def to_string(df):
     for col in df.columns:
         if df[col].dtype in ['int64','float','object']:
                df[col] = df[col].astype(str)
     return df
    # changing all columns to strings
    df_1 = to_string(movies_df)
    # joining director names 
    df_1['director'] = df_1['director'].apply(lambda x: "".join(x.lower() for x in x.split()))
    # removing pipes, joining title cast and returning first 3 names
    df_1['title_cast'] = df_1['title_cast'].apply(lambda x: "".join(x.lower() for x in x.split()))
    df_1['title_cast'] = df_1['title_cast'].map(lambda x: x.split('|')[:3])
    # removing pipes, joining plot keywords and returning first 5
    df_1['plot_keywords'] = df_1['plot_keywords'].map(lambda x: x.split('|')[:5])
    df_1['plot_keywords'] = df_1['plot_keywords'].apply(lambda x: " ".join(x))
    # Discarding the pipes between the genres 
    df_1['genres'] = df_1['genres'].map(lambda x: x.lower().split('|'))
    df_1['genres'] = df_1['genres'].apply(lambda x: " ".join(x))
    # setting movie titles as index
    df_1.set_index('title', inplace = True)
    # creating a new column  consisting of all our movie attributes
    df_1['KeyWords'] = ''
    columns = df_1.columns
    for index, row in df_1.iterrows():
         words = ''
         for col in columns:
             if col not in ['director','plot_keywords','genres']:
                 words = words + ' '.join(row[col])+ ' '
             else:
                 words = words + row[col]+ ' '
         row['KeyWords'] = words
    # resetting our index back to default index
    df_1.reset_index(inplace=True)

    # Subset of the data
    movies_subset = df_1[:subset_size]
    return movies_subset

# !! DO NOT CHANGE THIS FUNCTION SIGNATURE !!
# You are, however, encouraged to change its content.  
def content_model(movie_list,top_n=10):
    """Performs Content filtering based upon a list of movies supplied
       by the app user.

    Parameters
    ----------
    movie_list : list (str)
        Favorite movies chosen by the app user.
    top_n : type
        Number of top recommendations to return to the user.

    Returns
    -------
    list (str)
        Titles of the top-n movie recommendations to the user.

    """
    # Initializing the empty list of recommended movies
    recommended_movies = []
    data = data_preprocessing(40000) 
    # Instantiating and generating the count matrix
    count_vec = CountVectorizer()
    count_matrix = count_vec.fit_transform(data['KeyWords'])
    indices = pd.Series(data['title'])
    cosine_sim = cosine_similarity(count_matrix, count_matrix)
    # Getting the index of the movie that matches the title
    idx_1 = indices[indices == movie_list[0]].index[0]
    idx_2 = indices[indices == movie_list[1]].index[0]
    idx_3 = indices[indices == movie_list[2]].index[0]
    # Creating a Series with the similarity scores in descending order
    rank_1 = cosine_sim[idx_1]
    rank_2 = cosine_sim[idx_2]
    rank_3 = cosine_sim[idx_3]
    # Calculating the scores
    score_series_1 = pd.Series(rank_1).sort_values(ascending = False)
    score_series_2 = pd.Series(rank_2).sort_values(ascending = False)
    score_series_3 = pd.Series(rank_3).sort_values(ascending = False)
    # Getting the indexes of the 10 most similar movies
    listings = score_series_1.append(score_series_1).append(score_series_3).sort_values(ascending = False)

    # Store movie names
    recommended_movies = []
    # Appending the names of movies
    top_50_indexes = list(listings.iloc[1:50].index)
    # Removing chosen movies
    top_indexes = np.setdiff1d(top_50_indexes,[idx_1,idx_2,idx_3])
    for i in top_indexes[:top_n]:
        recommended_movies.append(list(movies['title'])[i])
    return recommended_movies
