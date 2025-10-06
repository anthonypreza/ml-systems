"""Preprocessing utils."""


def filter_interactions(interactions_df):
    core_interactions = interactions_df[
        [
            "user_id",
            "book_id",
            "rating",
            "is_read",
            "review_text_incomplete",
            "date_added",
        ]
    ]

    # Filter users and books with at least 3 interactions
    user_counts = core_interactions["user_id"].value_counts()
    book_counts = core_interactions["book_id"].value_counts()

    filtered_interactions = core_interactions[
        core_interactions["user_id"].isin(user_counts[user_counts >= 3].index)
        & core_interactions["book_id"].isin(book_counts[book_counts >= 3].index)
    ].copy()

    return filtered_interactions
