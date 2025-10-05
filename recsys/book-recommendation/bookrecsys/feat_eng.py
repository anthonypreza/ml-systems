"""Runs book recommendation feature engineering pipeline."""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from textblob import TextBlob
from .logger import get_logger


logger = get_logger("feat_eng")


def compute_genre_similarity(user_profile, book_genres):
    """Compute cosine similarity between user profile and book"""
    if not isinstance(book_genres, dict) or not book_genres:
        return 0

    # Get common genres
    common_genres = set(user_profile.keys()) & set(book_genres.keys())
    if not common_genres:
        return 0

    # Compute cosine similarity
    dot_product = sum(
        user_profile.get(genre, 0) * book_genres.get(genre, 0)
        for genre in common_genres
    )
    user_norm = sum(v**2 for v in user_profile.values()) ** 0.5
    book_norm = sum(v**2 for v in book_genres.values()) ** 0.5

    if user_norm == 0 or book_norm == 0:
        return 0

    return dot_product / (user_norm * book_norm)


def compute_year_preferences(filtered_interactions, books_df):
    user_book_years = filtered_interactions.merge(
        books_df[["book_id", "publication_year"]], on="book_id"
    )
    user_book_years["publication_year"] = pd.to_numeric(
        user_book_years["publication_year"], errors="coerce"
    )

    user_year_prefs = (
        user_book_years.groupby("user_id")["publication_year"]
        .agg(["mean", "std"])
        .fillna(0)
    )
    user_year_prefs.columns = ["user_year_pref_mean", "user_year_pref_std"]
    return user_year_prefs


def get_sentiment_polarity(text):
    """Get sentiment polarity (-1 to 1)"""
    if pd.isna(text) or len(str(text).strip()) == 0:
        return 0.0
    try:
        return TextBlob(str(text)).sentiment.polarity
    except:
        return 0.0


def run_feature_eng_pipeline(filtered_interactions, books_df, genres_df, authors_df):
    # Feature engineering
    logger.info("=== FEATURE ENGINEERING ===")
    # Add derived features to positive interactions
    filtered_interactions["is_read"] = filtered_interactions["is_read"].astype(int)
    # Skip rating, explicit_rating, review_length - these are outcomes of reading, not predictors

    # Compute user and book profiles from interactions only (no target leakage)
    user_features = filtered_interactions.groupby("user_id").agg(
        {"rating": ["mean", "std", "count"], "book_id": "nunique"}
    )

    book_features = filtered_interactions.groupby("book_id").agg(
        {"rating": ["mean", "std", "count"], "user_id": "nunique"}
    )

    # Flatten the MultiIndex columns
    user_features.columns = ["_".join(col).strip() for col in user_features.columns]
    book_features.columns = ["_".join(col).strip() for col in book_features.columns]

    # Reset index to make user_id/book_id regular columns
    user_features = user_features.reset_index()
    book_features = book_features.reset_index()

    # Create completion prediction dataset (only real interactions)
    completion_samples = filtered_interactions[
        [
            "user_id",
            "book_id",
            "is_read",  # TARGET: Will user complete this book?
            "date_added",  # Only temporal info, no interaction outcomes
        ]
    ].copy()

    logger.info("=== COMPLETION PREDICTION DATASET ===")
    logger.info("Dataset shape: %s", completion_samples.shape)
    logger.info("Target distribution:")
    logger.info("%s", completion_samples["is_read"].value_counts())
    logger.info("Completion rate: %.3f", completion_samples["is_read"].mean())

    train_data = completion_samples.copy()

    # No synthetic negatives needed - using real interactions only
    all_samples = train_data.copy()

    # Skip negative sample creation
    # all_samples = pd.concat([positive_samples, negative_samples], ignore_index=True)

    # Add user and book profiles to our completion dataset
    train_data = all_samples.merge(
        user_features, on="user_id", how="left", suffixes=("", "_user")
    )
    train_data = train_data.merge(
        book_features, on="book_id", how="left", suffixes=("", "_book")
    )

    # Skip negative sample creation - using only real interactions now
    train_data = all_samples.merge(
        user_features, on="user_id", how="left", suffixes=("", "_user")
    )
    train_data = train_data.merge(
        book_features, on="book_id", how="left", suffixes=("", "_book")
    )

    # Similarity features
    train_data["rating_diff"] = abs(
        train_data["rating_mean"] - train_data["rating_mean_book"]
    )
    train_data["rating_similarity"] = 1 / 1 + train_data["rating_diff"]

    user_languages = train_data.merge(
        books_df[["book_id", "language_code"]], on="book_id"
    )
    user_language_prefs = user_languages.groupby("user_id")["language_code"].apply(
        lambda x: x.value_counts().index[0] if len(x) > 0 else None
    )

    book_languages = books_df.set_index("book_id")["language_code"]

    train_data["user_preferred_lang"] = train_data["user_id"].map(user_language_prefs)

    train_data["book_language"] = train_data["book_id"].map(book_languages)
    train_data["language_match"] = (
        train_data["user_preferred_lang"] == train_data["book_language"]
    ).astype(int)

    user_books_with_genres = filtered_interactions.merge(
        genres_df, on="book_id", how="left"
    )
    train_data = train_data.merge(genres_df, on="book_id", how="left")

    user_genre_profiles = {}

    for user_id in user_books_with_genres["user_id"].unique():
        user_books = user_books_with_genres[
            user_books_with_genres["user_id"] == user_id
        ]
        user_profile = {}

        for _, row in user_books.iterrows():
            genres_dict = row["genres"]
            if isinstance(genres_dict, dict) and len(genres_dict) > 0:
                # Normalize weights within each book
                total_weight = sum(genres_dict.values())
                for genre, weight in genres_dict.items():
                    normalized_weight = weight / total_weight if total_weight > 0 else 0
                    user_profile[genre] = user_profile.get(genre, 0) + normalized_weight

        user_genre_profiles[user_id] = user_profile

    # Now compute similarities with the corrected profiles
    def get_genre_similarity_final(row):
        user_id = row["user_id"]
        book_genres = row["genres"]

        # Get user profile
        user_profile = user_genre_profiles.get(user_id, {})

        # Handle NaN book genres
        if pd.isna(book_genres) or not isinstance(book_genres, dict):
            return 0.0

        # Compute similarity
        return compute_genre_similarity(user_profile, book_genres)

    train_data["genre_similarity"] = train_data.apply(
        get_genre_similarity_final, axis=1
    )

    train_data = train_data.merge(
        books_df[["book_id", "publication_year"]], on="book_id", how="left"
    )

    # Add book format features
    train_data = train_data.merge(
        books_df[["book_id", "is_ebook", "format"]], on="book_id", how="left"
    )

    # Clean and encode is_ebook
    train_data["is_ebook"] = (
        train_data["is_ebook"]
        .map({"true": 1, "True": 1, True: 1, "false": 0, "False": 0, False: 0})
        .fillna(0)
        .astype(int)
    )

    # Get user ebook preferences
    user_ebook_preference = filtered_interactions.merge(
        books_df[["book_id", "is_ebook"]], on="book_id"
    )
    user_ebook_preference["is_ebook"] = (
        user_ebook_preference["is_ebook"]
        .map({"true": 1, "True": 1, True: 1, "false": 0, "False": 0, False: 0})
        .fillna(0)
        .astype(int)
    )

    user_ebook_pref = user_ebook_preference.groupby("user_id")["is_ebook"].mean()
    train_data["user_ebook_preference"] = (
        train_data["user_id"].map(user_ebook_pref).fillna(0.5)
    )

    # Ebook preference match
    train_data["ebook_preference_match"] = 1 - abs(
        train_data["user_ebook_preference"] - train_data["is_ebook"]
    )

    # Format feature engineering
    logger.info("=== FORMAT FEATURE ENGINEERING ===")

    # Check format distribution
    logger.info("Format value counts:")
    format_counts = books_df["format"].value_counts().head(10)
    logger.info("%s", format_counts)

    # Clean format data - handle missing and normalize
    train_data["format"] = train_data["format"].fillna("Unknown")

    # Create format categories

    def categorize_format(format_val):
        if pd.isna(format_val) or format_val == "" or format_val == "Unknown":
            return "Unknown"
        elif "Paperback" in str(format_val):
            return "Paperback"
        elif "Hardcover" in str(format_val) or "Hardback" in str(format_val):
            return "Hardcover"
        elif (
            "Kindle" in str(format_val)
            or "ebook" in str(format_val)
            or "eBook" in str(format_val)
        ):
            return "Digital"
        elif "Mass Market" in str(format_val):
            return "Mass Market"
        elif "Audio" in str(format_val):
            return "Audio"
        else:
            return "Other"

    train_data["format_category"] = train_data["format"].apply(categorize_format)

    logger.info("\nFormat categories:")
    logger.info("%s", train_data["format_category"].value_counts())

    # One-hot encode format categories (for top categories)
    format_dummies = pd.get_dummies(
        train_data["format_category"], prefix="format"
    ).astype(int)
    train_data = pd.concat([train_data, format_dummies], axis=1)

    # Ensure boolean dummies are 0/1 ints
    for _c in [
        c
        for c in train_data.columns
        if c.startswith("format_") and c != "format_category"
    ]:
        train_data[_c] = train_data[_c].astype(int)

    # Get user format preferences
    user_format_data = filtered_interactions.merge(
        books_df[["book_id", "format"]], on="book_id"
    )
    user_format_data["format_category"] = user_format_data["format"].apply(
        categorize_format
    )

    # Calculate user's preferred format (most common format they interact with)
    user_format_prefs = user_format_data.groupby("user_id")["format_category"].apply(
        lambda x: x.value_counts().index[0] if len(x) > 0 else "Unknown"
    )

    # Add user format preference match
    train_data["user_preferred_format"] = (
        train_data["user_id"].map(user_format_prefs).fillna("Unknown")
    )
    train_data["format_preference_match"] = (
        train_data["user_preferred_format"] == train_data["format_category"]
    ).astype(int)

    logger.info(
        "✅ Added format features: %d one-hot encoded categories",
        len(format_dummies.columns),
    )

    train_data["publication_year"] = pd.to_numeric(
        train_data["publication_year"], errors="coerce"
    )

    user_year_prefs = compute_year_preferences(filtered_interactions, books_df)
    train_data = train_data.merge(
        user_year_prefs, on="user_id", how="left", suffixes=("", "_user_pref")
    )

    # === AUTHOR-BASED FEATURES ===
    logger.info("=== AUTHOR FEATURE ENGINEERING ===")

    def extract_primary_author_id(authors_field):
        try:
            if isinstance(authors_field, list) and len(authors_field) > 0:
                a0 = authors_field[0]
                if isinstance(a0, dict) and "author_id" in a0:
                    return str(a0.get("author_id"))
        except (ValueError, RuntimeError):
            pass
        return None

    # Map each book to a primary author id
    books_author_map = books_df[["book_id", "authors"]].copy()
    books_author_map["primary_author_id"] = books_author_map["authors"].apply(
        extract_primary_author_id
    )

    # Bring primary author onto train_data
    train_data = train_data.merge(
        books_author_map[["book_id", "primary_author_id"]], on="book_id", how="left"
    )

    # Prepare author stats
    author_cols = [
        "author_id",
        "average_rating",
        "ratings_count",
        "text_reviews_count",
        "name",
    ]
    authors_stats = authors_df[author_cols].copy()
    authors_stats["author_id"] = authors_stats["author_id"].astype(str)
    for c in ["average_rating", "ratings_count", "text_reviews_count"]:
        authors_stats[c] = pd.to_numeric(authors_stats[c], errors="coerce").fillna(0)

    # Merge author stats to training data
    train_data = train_data.merge(
        authors_stats.rename(
            columns={
                "author_id": "primary_author_id",
                "average_rating": "author_avg_rating",
                "ratings_count": "author_ratings_count",
                "text_reviews_count": "author_text_reviews_count",
                "name": "author_name",
            }
        ),
        on="primary_author_id",
        how="left",
    )

    # User–author familiarity signals
    ua = filtered_interactions.merge(
        books_author_map[["book_id", "primary_author_id"]], on="book_id", how="left"
    )
    ua_counts = (
        ua.groupby(["user_id", "primary_author_id"])
        .size()
        .rename("user_author_interactions")
    )
    user_total = ua.groupby("user_id").size().rename("user_total_interactions")

    tmp = train_data[["user_id", "primary_author_id"]].copy()
    tmp = tmp.merge(
        ua_counts.reset_index(), on=["user_id", "primary_author_id"], how="left"
    )
    tmp = tmp.merge(user_total.reset_index(), on="user_id", how="left")
    train_data["user_author_interaction_count"] = (
        tmp["user_author_interactions"].fillna(0).astype(float)
    )
    train_data["user_author_interaction_ratio"] = (
        (
            train_data["user_author_interaction_count"]
            / tmp["user_total_interactions"].replace({0: pd.NA})
        )
        .fillna(0)
        .astype(float)
    )

    # Author popularity signals
    train_data["author_popularity_log"] = np.log1p(
        train_data["author_ratings_count"].fillna(0)
    )

    logger.info(
        "✅ Added author features: avg_rating, popularity, and user-author familiarity"
    )
    # Simple flag: has the user read this author before?
    train_data["has_read_author_before"] = (
        train_data["user_author_interaction_count"] > 0
    ).astype(int)

    # === TEXT SIMILARITY FEATURES (REVIEW TEXT + TITLE) ===
    logger.info("=== TEXT SIMILARITY FEATURES ===")

    # Build corpora from interactions
    reviews = filtered_interactions[
        ["user_id", "book_id", "review_text_incomplete"]
    ].dropna()
    reviews["review_text_incomplete"] = reviews["review_text_incomplete"].astype(str)

    # Aggregate user and book review text
    user_review_text = (
        reviews.groupby("user_id")["review_text_incomplete"]
        .apply(lambda x: " ".join(x.astype(str)))
        .reset_index()
    )
    book_review_text = (
        reviews.groupby("book_id")["review_text_incomplete"]
        .apply(lambda x: " ".join(x.astype(str)))
        .reset_index()
    )

    # Fit a single TF-IDF on combined corpus to align spaces
    combined_corpus = pd.concat(
        [
            user_review_text["review_text_incomplete"],
            book_review_text["review_text_incomplete"],
        ],
        ignore_index=True,
    )

    if len(combined_corpus) > 0:
        tfidf_reviews = TfidfVectorizer(
            stop_words="english", max_features=5000, min_df=5
        )
        tfidf_reviews.fit(combined_corpus)

        U_rev = normalize(
            tfidf_reviews.transform(user_review_text["review_text_incomplete"])
        )
        B_rev = normalize(
            tfidf_reviews.transform(book_review_text["review_text_incomplete"])
        )

        # Build lookup dicts of normalized sparse vectors
        user_ids = user_review_text["user_id"].tolist()
        book_ids = book_review_text["book_id"].tolist()
        user_rev_vec = {uid: U_rev[i] for i, uid in enumerate(user_ids)}
        book_rev_vec = {bid: B_rev[i] for i, bid in enumerate(book_ids)}

        # Compute cosine similarity (dot of normalized vectors)
        def review_text_similarity(row):
            u = user_rev_vec.get(row["user_id"])
            b = book_rev_vec.get(row["book_id"])
            if u is None or b is None:
                return 0.0
            return float(u.multiply(b).sum())

        train_data["review_text_similarity"] = train_data.apply(
            review_text_similarity, axis=1
        )
    else:
        train_data["review_text_similarity"] = 0.0

    logger.info("✅ Added review_text_similarity")

    # Title similarity (word n-grams)
    book_titles = books_df[["book_id", "title"]].dropna().copy()
    book_titles["title"] = book_titles["title"].astype(str)
    user_titles = filtered_interactions.merge(book_titles, on="book_id", how="left")
    user_titles = user_titles.dropna(subset=["title"])
    user_title_agg = (
        user_titles.groupby("user_id")["title"]
        .apply(lambda x: " ".join(x.astype(str)))
        .reset_index()
    )

    combined_titles = pd.concat(
        [user_title_agg["title"], book_titles["title"]], ignore_index=True
    )

    if len(combined_titles) > 0:
        tfidf_titles = TfidfVectorizer(
            stop_words="english", ngram_range=(1, 2), max_features=3000, min_df=3
        )
        tfidf_titles.fit(combined_titles)

        U_tit = normalize(tfidf_titles.transform(user_title_agg["title"]))
        B_tit = normalize(tfidf_titles.transform(book_titles["title"]))

        user_tit_vec = {
            uid: U_tit[i] for i, uid in enumerate(user_title_agg["user_id"].tolist())
        }
        book_tit_vec = {
            bid: B_tit[i] for i, bid in enumerate(book_titles["book_id"].tolist())
        }

        def title_text_similarity(row):
            u = user_tit_vec.get(row["user_id"])
            b = book_tit_vec.get(row["book_id"])
            if u is None or b is None:
                return 0.0
            return float(u.multiply(b).sum())

        train_data["title_similarity"] = train_data.apply(title_text_similarity, axis=1)
    else:
        train_data["title_similarity"] = 0.0

    logger.info("✅ Added title_similarity")

    train_data["year_diff"] = abs(
        train_data["publication_year"] - train_data["user_year_pref_mean"]
    )
    train_data["year_similarity"] = np.exp(
        -train_data["year_diff"] / (train_data["user_year_pref_std"] + 1)
    )

    book_length_dict = pd.to_numeric(
        books_df.set_index("book_id")["num_pages"], errors="coerce"
    )
    train_data["book_length"] = train_data["book_id"].map(book_length_dict)

    user_length_prefs = filtered_interactions.merge(
        books_df[["book_id", "num_pages"]], on="book_id"
    )
    user_length_prefs["num_pages"] = pd.to_numeric(
        user_length_prefs["num_pages"], errors="coerce"
    )
    user_avg_length = user_length_prefs.groupby("user_id")["num_pages"].mean()

    train_data["user_avg_book_length"] = train_data["user_id"].map(user_avg_length)
    train_data["length_similarity"] = 1 / (
        1 + abs(train_data["book_length"] - train_data["user_avg_book_length"]) / 100
    )

    # Reading frequency patterns
    train_data["date_added"] = pd.to_datetime(
        train_data["date_added"], format="%Y-%m-%d %H:%M:%S%z", errors="coerce"
    )
    train_data["hour_added"] = train_data["date_added"].dt.hour
    train_data["day_of_week"] = train_data["date_added"].dt.dayofweek
    train_data["month_added"] = train_data["date_added"].dt.month

    # TODO: Explore seasonal patterns at some point, for now this approach doesn' work
    # # Seasonal reading patterns - simpler approach
    # user_seasonal_patterns = filtered_interactions.copy()
    # user_seasonal_patterns['date_added'] = pd.to_datetime(user_seasonal_patterns['date_added'], format='%a %b %d %H:%M:%S %z %Y', errors='coerce')

    # # Create season column with NaN for invalid dates
    # user_seasonal_patterns['season'] = user_seasonal_patterns['date_added'].dt.month % 12 // 3

    # # User's preferred reading season (dropna handles NaN values automatically)
    # user_season_prefs = user_seasonal_patterns.dropna(subset=['season']).groupby('user_id')['season'].apply(
    #     lambda x: x.value_counts().index[0] if len(x) > 0 else None,
    #     include_groups=False
    # )

    # train_data['book_season'] = train_data['month_added'] % 12 // 3
    # train_data['user_preferred_season'] = train_data['user_id'].map(user_season_prefs)
    # train_data['season_match'] = (train_data['user_preferred_season'] == train_data['book_season']).astype(int)

    # TODO: FUTURE IMPROVEMENT - Add scalable leave-one-out completion rates
    # Current challenge: Preventing target leakage while maintaining computational efficiency
    #
    # COMMENTED OUT: Computationally expensive leave-one-out approach
    # - User completion rate (excluding current book)
    # - Book completion rate (excluding current user)
    #
    # Potential optimizations for future implementation:
    # 1. Pre-compute global completion rates and subtract current interaction
    # 2. Use vectorized operations instead of apply()
    # 3. Implement efficient caching for repeated calculations
    # 4. Consider approximate methods (e.g., sample-based estimates)
    #
    # For now, skipping these features to maintain training speed and scalability

    # def compute_user_completion_rate_loo(row):
    #     """Compute user completion rate excluding current book (leave-one-out)"""
    #     user_id = row['user_id']
    #     book_id = row['book_id']
    #
    #     # Get all interactions for this user EXCEPT current book
    #     user_other_books = filtered_interactions[
    #         (filtered_interactions['user_id'] == user_id) &
    #         (filtered_interactions['book_id'] != book_id)
    #     ]
    #
    #     if len(user_other_books) == 0:
    #         return 0.5  # Default if no other books (neutral prior)
    #
    #     return user_other_books['is_read'].mean()

    # logger.info("Computing leave-one-out user completion rates (no target leakage)...")
    # train_data["user_completion_rate"] = train_data.apply(compute_user_completion_rate_loo, axis=1)

    # def compute_book_completion_rate_loo(row):
    #     """Compute book completion rate excluding current user (leave-one-out)"""
    #     user_id = row['user_id']
    #     book_id = row['book_id']
    #
    #     # Get all interactions for this book EXCEPT current user
    #     book_other_users = filtered_interactions[
    #         (filtered_interactions['book_id'] == book_id) &
    #         (filtered_interactions['user_id'] != user_id)
    #     ]
    #
    #     if len(book_other_users) == 0:
    #         return 0.5  # Default if no other users (neutral prior)
    #
    #     return book_other_users['is_read'].mean()

    # logger.info("Computing leave-one-out book completion rates (no target leakage)...")
    # train_data["book_typical_completion"] = train_data.apply(compute_book_completion_rate_loo, axis=1)

    # train_data["completion_alignment"] = 1 - abs(
    #     train_data["user_completion_rate"] - train_data["book_typical_completion"]
    # )

    logger.info(
        "Skipped completion rate features to avoid target leakage and maintain scalability"
    )

    # User sentiment patterns
    user_sentiment_profiles = filtered_interactions.groupby("user_id")[
        "review_text_incomplete"
    ].apply(
        lambda reviews: np.mean([get_sentiment_polarity(review) for review in reviews])
    )

    # Book sentiment patterns
    book_sentiment_profiles = filtered_interactions.groupby("book_id")[
        "review_text_incomplete"
    ].apply(
        lambda reviews: np.mean([get_sentiment_polarity(review) for review in reviews])
    )

    # Add sentiment similarity
    train_data["user_avg_sentiment"] = train_data["user_id"].map(
        user_sentiment_profiles
    )
    train_data["book_avg_sentiment"] = train_data["book_id"].map(
        book_sentiment_profiles
    )
    train_data["sentiment_similarity"] = (
        1 - abs(train_data["user_avg_sentiment"] - train_data["book_avg_sentiment"]) / 2
    )

    # Historical rating patterns (no leakage - uses OTHER books user has rated)
    logger.info("=== HISTORICAL RATING PATTERNS ===")

    # TODO: FUTURE IMPROVEMENT - Add temporal filtering
    # Currently using all ratings from user's history and others' ratings of books
    # For production, should only use ratings that occurred BEFORE current interaction
    # This prevents future information leakage but adds complexity

    # User's rating behavior on OTHER books (not current one)

    def compute_user_rating_history(interactions):
        """Compute user rating patterns from their history (excluding current book)"""
        user_rating_patterns = {}

        for user_id in interactions["user_id"].unique():
            user_books = interactions[interactions["user_id"] == user_id]

            if len(user_books) > 1:  # Need multiple books for patterns
                # Only use ratings (NOT is_read which is our target)
                rated_books = user_books[user_books["rating"] > 0]

                if len(rated_books) > 0:
                    user_rating_patterns[user_id] = {
                        "user_avg_rating_historical": rated_books["rating"].mean(),
                        "user_rating_std_historical": rated_books["rating"].std()
                        if len(rated_books) > 1
                        else 0,
                        "user_harsh_rater": (rated_books["rating"] <= 2).mean(),
                        "user_generous_rater": (rated_books["rating"] >= 4).mean(),
                        "user_rating_count_historical": len(rated_books),
                    }

        return user_rating_patterns

    # Compute historical patterns
    user_rating_history = compute_user_rating_history(filtered_interactions)

    # Add to training data (only rating-based features, NO is_read features)
    for feature in [
        "user_avg_rating_historical",
        "user_rating_std_historical",
        "user_harsh_rater",
        "user_generous_rater",
        "user_rating_count_historical",
    ]:
        train_data[feature] = train_data["user_id"].map(
            lambda uid: user_rating_history.get(uid, {}).get(feature, 0)
        )

    logger.info("✅ Added historical rating features (no target leakage)")
    logger.info("⚠️  Note: Temporal leakage possible - future improvement needed")
    logger.info("Users with rating history: %d", len(user_rating_history))

    # Book shelf features
    from collections import Counter

    def count_shelves_in_list(shelves):
        c = Counter()
        for s in shelves:
            c[s["name"]] += int(s["count"])
        return c

    # Global shelf popularity across all books
    global_shelf_counts = Counter()
    for shelves in books_df["popular_shelves"]:
        global_shelf_counts.update(count_shelves_in_list(shelves))

    TOP_N = 100  # tune as you like (50/100/200)
    top_shelves = [name for name, _ in global_shelf_counts.most_common(TOP_N)]

    def compute_total_shelf_count(popular_shelves: list):
        cnt = 0
        for shelf in popular_shelves:
            cnt += int(shelf["count"])
        return cnt

    def compute_top_n_shelf_props(popular_shelves: list, top_shelves=top_shelves):
        # total count for this book
        total = sum(int(s["count"]) for s in popular_shelves)
        if total == 0:
            return {shelf: 0.0 for shelf in top_shelves}

        # build dict for this book
        counts = {s["name"]: int(s["count"]) for s in popular_shelves}
        return {shelf: counts.get(shelf, 0) / total for shelf in top_shelves}

    shelves_df = books_df[["book_id", "popular_shelves"]].copy()
    shelves_df["total_shelf_count"] = shelves_df["popular_shelves"].apply(
        compute_total_shelf_count
    )

    # compute proportions dict
    shelves_df["shelf_props"] = shelves_df["popular_shelves"].apply(
        compute_top_n_shelf_props
    )

    # expand dict into separate columns
    shelf_prop_df = shelves_df["shelf_props"].apply(pd.Series)

    # merge back
    shelves_df = pd.concat(
        [shelves_df[["book_id", "total_shelf_count"]], shelf_prop_df], axis=1
    )

    # User shelf features

    user_shelves_df = filtered_interactions[["user_id", "book_id"]].merge(
        books_df[["book_id", "popular_shelves"]], on="book_id"
    )

    def aggregate_user_shelves(shelf_lists):
        counter = Counter()
        for shelves in shelf_lists:
            for s in shelves:
                counter[s["name"]] += int(s["count"])
        return dict(counter)

    user_shelf_profiles = user_shelves_df.groupby("user_id")["popular_shelves"].apply(
        aggregate_user_shelves
    )

    user_shelf_profiles_wide = user_shelf_profiles.unstack(fill_value=0)

    # restrict/reorder user shelves to the same top_shelves
    user_shelf_profiles_aligned = user_shelf_profiles_wide.reindex(
        columns=top_shelves, fill_value=0
    )

    # normalize to proportions
    user_shelf_profiles_aligned = user_shelf_profiles_aligned.div(
        user_shelf_profiles_aligned.sum(axis=1).replace(0, 1), axis=0
    )

    def cosine_sim(u, v):
        nu, nv = np.linalg.norm(u), np.linalg.norm(v)
        if nu == 0 or nv == 0:
            return 0.0
        return float(np.dot(u, v) / (nu * nv))

    user_profiles = user_shelf_profiles_aligned.reset_index().rename(
        columns={s: s + "_user" for s in top_shelves}
    )
    book_profiles = shelves_df[["book_id"] + top_shelves].rename(
        columns={s: s + "_book" for s in top_shelves}
    )

    user_profiles = user_profiles.fillna(0.0)
    book_profiles = book_profiles.fillna(0.0)

    interactions_with_sim = (
        filtered_interactions[["user_id", "book_id"]]
        .merge(user_profiles, on="user_id", how="left")
        .merge(book_profiles, on="book_id", how="left")
    )

    u_cols = [s + "_user" for s in top_shelves]
    b_cols = [s + "_book" for s in top_shelves]

    # user and book matrices
    U = interactions_with_sim[u_cols].to_numpy(dtype=float)
    B = interactions_with_sim[b_cols].to_numpy(dtype=float)

    # dot product row-wise
    dot = np.einsum("ij,ij->i", U, B)

    # norms
    U_norm = np.linalg.norm(U, axis=1)
    B_norm = np.linalg.norm(B, axis=1)

    # avoid div by 0
    denom = U_norm * B_norm
    denom[denom == 0] = 1.0

    # cosine similarity
    cos_sim = dot / denom

    # assign back
    interactions_with_sim["shelf_cosine_similarity"] = cos_sim
    interactions_with_sim[u_cols + b_cols] = interactions_with_sim[
        u_cols + b_cols
    ].fillna(0.0)

    train_data = train_data.merge(interactions_with_sim, on=["user_id", "book_id"])

    train_data["shelf_cosine_similarity"] = train_data[
        "shelf_cosine_similarity"
    ].fillna(0.0)

    # the following features are dropped as their representations are already captured
    # Keep user_id and book_id in the exported dataset so downstream
    # services can join metadata and power serving.
    features_to_drop_for_ml = [
        "genres",
        "date_added",
        "user_preferred_lang",
        "book_language",
        "format",
        "format_category",
        "user_preferred_format",
        "primary_author_id",
        "author_name",
    ]
    train_data = train_data.drop(columns=features_to_drop_for_ml)

    # No null interaction features to fill - we didn't create leaky features
    # Only need to handle missing values in legitimate features

    # Fill temporal features
    datetime_cols = ["hour_added", "day_of_week", "month_added"]
    train_data[datetime_cols] = train_data[datetime_cols].fillna(0)

    # Similarity features (0 = neutral/unknown)
    similarity_cols = [
        "year_similarity",
        "length_similarity",
        "sentiment_similarity",
        "genre_similarity",
        "review_text_similarity",
        "title_similarity",
    ]
    for col in similarity_cols:
        train_data[col] = train_data[col].fillna(0.0)

    # Fill remaining nulls for year and length-based features
    logger.info("=== Filling remaining nulls for year/length features ===")

    # Ensure publication_year is numeric
    train_data["publication_year"] = pd.to_numeric(
        train_data["publication_year"], errors="coerce"
    )

    # Compute robust medians
    med_year = train_data["publication_year"].median()
    med_len = train_data["book_length"].median()
    med_user_year_std = train_data["user_year_pref_std"].median()

    # Fill NaNs
    train_data["publication_year"] = train_data["publication_year"].fillna(med_year)
    train_data["user_year_pref_mean"] = train_data["user_year_pref_mean"].fillna(
        med_year
    )
    train_data["user_year_pref_std"] = train_data["user_year_pref_std"].fillna(
        med_user_year_std if pd.notnull(med_user_year_std) else 0.0
    )
    train_data["book_length"] = train_data["book_length"].fillna(med_len)
    train_data["user_avg_book_length"] = train_data["user_avg_book_length"].fillna(
        med_len
    )

    # Recompute dependent features
    if "year_diff" in train_data.columns:
        train_data["year_diff"] = (
            train_data["publication_year"] - train_data["user_year_pref_mean"]
        )
        train_data["year_similarity"] = 1 - abs(train_data["year_diff"]) / (
            train_data["user_year_pref_std"] + 1
        )
    if "length_similarity" in train_data.columns:
        train_data["length_similarity"] = 1 / (
            1
            + abs(train_data["book_length"] - train_data["user_avg_book_length"]) / 100
        )

    # Fill author-related features
    for col, val in {
        "author_avg_rating": 0.0,
        "author_ratings_count": 0.0,
        "author_text_reviews_count": 0.0,
        "author_popularity_log": 0.0,
        "user_author_interaction_count": 0.0,
        "user_author_interaction_ratio": 0.0,
        "has_read_author_before": 0,
    }.items():
        if col in train_data.columns:
            train_data[col] = train_data[col].fillna(val)

    # Fill user profile features (removed completion rate features to avoid leakage)
    user_profile_cols = ["rating_mean", "rating_std", "rating_count", "book_id_nunique"]

    for col in user_profile_cols:
        train_data[col].fillna(train_data[col].median(), inplace=True)

    # Recalculate dependent features
    train_data["rating_diff"] = abs(
        train_data["rating_mean"] - train_data["rating_mean_book"]
    )
    train_data["rating_similarity"] = 1 / (1 + train_data["rating_diff"])

    # Fix year features
    train_data["year_diff"].fillna(train_data["year_diff"].median(), inplace=True)

    # Verify no nulls remain
    logger.info("Remaining nulls:")
    logger.info("%s", train_data.isnull().sum()[train_data.isnull().sum() > 0])

    # Final check
    logger.info("Total nulls remaining: %d", int(train_data.isnull().sum().sum()))

    # After dropping raw columns, verify everything is numeric
    logger.info("Final feature types:")
    logger.info("%s", train_data.dtypes)

    # Should only see int64, float64, bool - no object/dict types
    non_numeric = train_data.select_dtypes(exclude=["number", "bool"]).columns
    if len(non_numeric) > 0:
        logger.info("Warning: Non-numeric columns remaining: %s", non_numeric.tolist())

    # Update user mapping for completion prediction

    user_book_mapping = all_samples[["user_id", "book_id"]].copy()
    user_book_mapping["sample_index"] = range(len(user_book_mapping))

    return train_data, user_book_mapping
