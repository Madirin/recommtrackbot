
# Development and Training Report for the Recommendation System

## Project Stages

1. **Task Formulation**
    
    - The goal of the project was to develop a Telegram bot for recommending music tracks based on user preferences.
    - The functionality included viewing recommendations, rating tracks and genres, and generating new recommendations based on the trained model.
2. **Data Collection and Preparation**
    
    - The dataset of music tracks was sourced from Spotify's library.
    - Libraries such as `pandas` and `sqlite` were used for data processing.
    - Data preprocessing steps:
    
    ```python
    # Removing unnecessary columns
    columns_to_drop = ['energy', 'danceability', 'explicit', 'duration_ms']
    df = df.drop(columns=columns_to_drop)
    
    # Cleaning data and removing rows without genres and artists
    df['genre'] = df['genre'].apply(clean_genre)
    df = df.dropna(subset=['genre'])
    
    df['artist_name'] = df['artist_name'].apply(clean_artist)
    df = df.dropna(subset=['artist_name'])
    
    df = (
        df
        .sort_values('normalized_popularity', ascending=False)
        .drop_duplicates(subset=['track_name', 'artist_name'], keep='first')
        .reset_index(drop=True)
    )
    ```
    
    - Removing duplicates:
    
    ```python
    df = df.drop_duplicates(subset='track_id')
    ```
    
    - Normalizing a numerical feature (popularity):
    
    ```python
    scaler = MinMaxScaler(feature_range=(0, 1))
    df['normalized_popularity'] = scaler.fit_transform(df[['popularity']])
    df = df[df['normalized_popularity'] > 0]
    ```
    
    - Encoding categorical features (genres):
    
    ```python
    genre_popularity = df.groupby('genre')['normalized_popularity'].mean().reset_index().sort_values('normalized_popularity')
    
    artist_popularity = df.groupby('artist_name')['normalized_popularity'].mean().reset_index().sort_values('normalized_popularity')
    
    track_popularity = df.groupby('track_name')['normalized_popularity'].mean().reset_index().sort_values('normalized_popularity')
    
    artist_numeric_mapping = {artist: idx + 1 for idx, artist in enumerate(artist_popularity['artist_name'])}
    
    genre_numeric_mapping = {genre: idx + 1 for idx, genre in enumerate(genre_popularity['genre'])}
    
    track_numeric_mapping = {track: idx + 1 for idx, track in enumerate(track_popularity['track_name'])}
    ```
    
3. **Machine Learning Model Implementation**
    
    - Model: `RandomForestRegressor`
4. **Model Evaluation**
    
    - Metrics: `RMSE, PRECISION@K, Recall@K`
5. **Interface**
    
    - Platform: Telegram

## Methods Used

1. **Machine Learning Model**
    
    - `RandomForestRegressor` as an ensemble method:
        - `n_estimators=100`: The model builds **100 decision trees**, each trained on a random subset of data.
        - `random_state = 42`: Fixed initial value for the random number generator; any seed value can be used.
    - Splitting data into training and test sets in a 70/30 ratio:
        
        ```python
        # Example
        train_df = user_df.sample(frac=0.7, random_state=42)
        test_df = user_df.drop(train_df.index)
        ```
        
2. **Recommendation Implementation**
    
    - Recommendations for a user were generated based on their previous ratings.

## Interface

- The interface was built on a TelegramBot using the `python-telegram-bot` library, which provided a convenient interface to work with TelegramAPI.
- This library enabled processing user messages and creating menus with buttons.
- Its asynchronous capabilities contributed to high performance, making interactions with the bot fast and user-friendly.

## Data & Libraries

- The `sqlite` library was used for managing a local database to store user information, their ratings, and recommendation results. The library supports operations such as deleting, adding, and updating data.
- The `pandas` library was used to create a user `dataframe`, which was then used for model training and track recommendation.
- The `ast` library was employed for parsing strings containing genre and artist columns in the dataset.
- The `sklearn` library served as the primary library for creating and training the model.

## Results

- After training, the model demonstrated the following results based on experiments with two users:
    - **User-1 ("Casual Listener")**
        - **RMSE** (1.086): Measures the average prediction error in the same units as the user rating (scale of 1 to 5). An RMSE of 1.086 indicates predictions deviate by ~1.1 points on average.
        - **Precision@5** (0.8): The proportion of tracks in the top-5 recommendations that were relevant (user rating ≥ 4). A score of 0.8 means 80% of the top-5 tracks matched user preferences.
        - **Recall@5** (0.4): The proportion of all relevant tracks captured in the top-5 recommendations. A score of 0.4 indicates that only 40% of relevant tracks were included.
    - **User-2 ("Expert Listener")**
        - RMSE: 1.600960649110402
        - Precision@5: 0.4
        - Recall@5: 0.5

### Final Analysis

- **Advantages**:
    - The model performed well in terms of recommendation precision (Precision@5 = 80%).
    - RMSE indicated reasonable predictions (~1.1 error on a 1–5 scale).
- **Issues**:
    - Recall@5 = 0.4 reveals the model captures only 40% of all relevant tracks, highlighting the need for improving recommendation diversity and coverage.
- **User Analysis**:
    - Two users participated in the experiment. The first user rated tracks casually without much deliberation.
    - The second user, an expert, assigned ratings based on their extensive musical knowledge and preferences.
    - Below are the metrics plots of two users (based on `matplotlib`):

		![1](https://www.upload.ee/image/17570719/Pasted_image_20241229195346.png)
		![2](https://www.upload.ee/image/17570724/Pasted_image_20241229195353.png)
		![3](https://www.upload.ee/image/17570726/Pasted_image_20241229195406.png)

## Conclusion

- The Telegram bot was successfully implemented, providing an intuitive interface and high-quality recommendations.
- While the system demonstrated decent metric scores, it did not fully account for all user preferences (as seen in Recall@5).
- Future improvements could involve enhancing the model with a larger dataset, adding functionality for personalization, and incorporating neural networks.