"""Data loading and processing utilities for similar listings recommendation."""

import json
import uuid
import random
from collections import defaultdict
from typing import List, Dict, Any, Optional, Set, Tuple
from ..core.config import Config


# Data Loading Functions
def load_interactions(filepath: Optional[str] = None) -> List[Dict[str, Any]]:
    """Load interactions from JSON file."""
    if filepath is None:
        filepath = Config.INTERACTIONS_FILE

    with open(filepath, "r") as f:
        return json.load(f)


def load_sessions(filepath: Optional[str] = None) -> List[Dict[str, Any]]:
    """Load sessions from JSON file."""
    if filepath is None:
        filepath = Config.SESSIONS_FILE

    with open(filepath, "r") as f:
        return json.load(f)


def load_listings(filepath: Optional[str] = None) -> List[Dict[str, Any]]:
    """Load listings from JSON file."""
    if filepath is None:
        filepath = Config.LISTINGS_FILE

    with open(filepath, "r") as f:
        return json.load(f)


def load_triplets(filepath: Optional[str] = None) -> List[Dict[str, Any]]:
    """Load triplets from JSON file."""
    if filepath is None:
        filepath = Config.TRIPLETS_FILE

    with open(filepath, "r") as f:
        return json.load(f)


def load_users(filepath: Optional[str] = None) -> List[Dict[str, Any]]:
    """Load users from JSON file."""
    if filepath is None:
        filepath = Config.USERS_FILE

    with open(filepath, "r") as f:
        return json.load(f)


def load_all_data(data_dir: Optional[str] = None) -> Dict[str, List[Dict[str, Any]]]:
    """Load all data files."""
    if data_dir:
        # Override default paths with custom directory
        interactions_file = f"{data_dir}/interactions.json"
        sessions_file = f"{data_dir}/sessions.json"
        listings_file = f"{data_dir}/listings.json"
        triplets_file = f"{data_dir}/triplets.json"
        users_file = f"{data_dir}/users.json"
    else:
        interactions_file = None
        sessions_file = None
        listings_file = None
        triplets_file = None
        users_file = None

    return {
        "interactions": load_interactions(interactions_file),
        "sessions": load_sessions(sessions_file),
        "listings": load_listings(listings_file),
        "triplets": load_triplets(triplets_file),
        "users": load_users(users_file),
    }


def create_listing_mappings(
    triplets: List[Dict], listings: List[Dict]
) -> Tuple[Dict[str, int], Dict[int, str]]:
    """Create mappings between listing IDs and indices."""
    # Get all unique listing IDs from triplets and listings
    listing_ids = set()
    for triplet in triplets:
        listing_ids.add(triplet["central_listing"])
        listing_ids.add(triplet["context_listing"])

    for listing in listings:
        listing_ids.add(listing["id"])

    listing_ids = sorted(list(listing_ids))

    listing_to_idx = {lid: idx for idx, lid in enumerate(listing_ids)}
    idx_to_listing = {idx: lid for lid, idx in listing_to_idx.items()}

    return listing_to_idx, idx_to_listing


def split_triplets(
    triplets: List[Dict], train_ratio: Optional[float] = None
) -> Tuple[List[Dict], List[Dict]]:
    """Split triplets into train and validation sets."""
    if train_ratio is None:
        train_ratio = Config.TRAIN_RATIO

    random.shuffle(triplets)
    split_idx = int(len(triplets) * train_ratio)

    train_triplets = triplets[:split_idx]
    val_triplets = triplets[split_idx:]

    return train_triplets, val_triplets


def save_data(data: Any, filepath: str) -> None:
    """Save data to JSON file."""
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)


# Session Processing Classes
class SessionExtractor:
    """Extract search sessions from interaction data."""

    def __init__(self, max_session_gap: Optional[int] = None):
        """
        Initialize session extractor.

        Args:
            max_session_gap: Maximum time gap (seconds) between interactions in same session
        """
        if max_session_gap is None:
            max_session_gap = Config.MAX_SESSION_GAP
        self.max_session_gap = max_session_gap

    def extract_sessions(
        self, interactions: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Extract sessions from interactions.

        A session is defined as a sequence of clicks followed by a booking without interruption.
        """
        # Group interactions by user and sort by timestamp
        user_interactions = defaultdict(list)
        for interaction in interactions:
            user_interactions[interaction["user_id"]].append(interaction)

        # Sort each user's interactions by timestamp
        for user_id in user_interactions:
            user_interactions[user_id].sort(key=lambda x: x["timestamp"])

        sessions = []

        for user_id, user_interacts in user_interactions.items():
            current_session = []
            last_timestamp = None

            for interaction in user_interacts:
                timestamp = interaction["timestamp"]
                interaction_type = interaction["interaction_type"]
                listing_id = interaction["listing_id"]

                # Check if this interaction continues the current session
                if (
                    last_timestamp is not None
                    and timestamp - last_timestamp > self.max_session_gap
                ):
                    # Session gap too large, start new session
                    self._finalize_session(current_session, sessions)
                    current_session = []

                # Add interaction to current session
                current_session.append(
                    {
                        "listing_id": listing_id,
                        "interaction_type": interaction_type,
                        "timestamp": timestamp,
                    }
                )

                # If this is a booking, finalize the session
                if interaction_type == "book":
                    self._finalize_session(current_session, sessions)
                    current_session = []

                last_timestamp = timestamp

            # Finalize any remaining session (though it won't have a booking)
            if current_session:
                self._finalize_session(current_session, sessions)

        return sessions

    def _finalize_session(
        self, session_interactions: List[Dict[str, Any]], sessions: List[Dict[str, Any]]
    ) -> None:
        """
        Finalize a session and add it to the sessions list.

        Only sessions with at least one click and ending with a booking are valid.
        """
        if len(session_interactions) < 2:  # Need at least one click + one booking
            return

        # Check if session ends with a booking
        if session_interactions[-1]["interaction_type"] != "book":
            return

        # Extract clicked listings (all except the final booking)
        clicked_listings = []
        for interaction in session_interactions[:-1]:
            if interaction["interaction_type"] in ["click", "view"]:
                clicked_listings.append(interaction["listing_id"])

        # Skip if no clicks before booking
        if not clicked_listings:
            return

        booked_listing = session_interactions[-1]["listing_id"]

        # Create session record
        session = {
            "session_id": str(uuid.uuid4()),
            "clicked_listing_ids": clicked_listings,
            "booked_listing_id": booked_listing,
            "session_start": session_interactions[0]["timestamp"],
            "session_end": session_interactions[-1]["timestamp"],
            "num_clicks": len(clicked_listings),
            "session_duration": session_interactions[-1]["timestamp"]
            - session_interactions[0]["timestamp"],
        }

        sessions.append(session)


class TripletGenerator:
    """Generate training triplets from session data."""

    def __init__(
        self,
        window_size: Optional[int] = None,
        neg_sampling_ratio: Optional[int] = None,
        seed: Optional[int] = None,
    ):
        """
        Initialize triplet generator.

        Args:
            window_size: Size of sliding window for context extraction
            neg_sampling_ratio: Number of negative samples per positive sample
            seed: Random seed for reproducibility
        """
        if window_size is None:
            window_size = Config.WINDOW_SIZE
        if neg_sampling_ratio is None:
            neg_sampling_ratio = Config.NEG_SAMPLING_RATIO
        if seed is None:
            seed = Config.SEED

        self.window_size = window_size
        self.neg_sampling_ratio = neg_sampling_ratio
        random.seed(seed)

    def build_neighborhood_index(self, listings: List[Dict]) -> Dict[str, List[str]]:
        """Build index of listings by city for neighborhood-based negative sampling."""
        neighborhood_index = defaultdict(list)
        for listing in listings:
            city = listing["city"]
            listing_id = listing["id"]
            neighborhood_index[city].append(listing_id)

        return neighborhood_index

    def extract_sliding_window_pairs(
        self, session: Dict[str, Any]
    ) -> List[Tuple[str, str]]:
        """Extract positive pairs using sliding window approach."""
        clicked_ids = session["clicked_listing_ids"]
        booked_id = session["booked_listing_id"]

        # Combine clicked and booked for full sequence
        full_sequence = clicked_ids + [booked_id]

        pairs = []

        # Sliding window pairs
        for i, central_listing in enumerate(full_sequence):
            # Define context window
            start_idx = max(0, i - self.window_size // 2)
            end_idx = min(len(full_sequence), i + self.window_size // 2 + 1)

            # Extract context listings (excluding central listing)
            context_listings = []
            for j in range(start_idx, end_idx):
                if j != i:
                    context_listings.append(full_sequence[j])

            # Create positive pairs
            for context_listing in context_listings:
                pairs.append((central_listing, context_listing))

        # Hard positive: all clicked listings with eventually booked listing
        for clicked_id in clicked_ids:
            pairs.append((clicked_id, booked_id))
            pairs.append((booked_id, clicked_id))  # Bidirectional

        return pairs

    def generate_negative_samples(
        self,
        central_listing: str,
        positive_contexts: Set[str],
        all_listings: List[str],
        neighborhood_index: Dict[str, List[str]],
        listing_to_city: Dict[str, str],
    ) -> List[str]:
        """Generate negative samples for a central listing."""
        negatives = []
        central_city = listing_to_city.get(central_listing)

        # Random negatives (easy negatives)
        random_negatives = []
        attempts = 0
        while len(random_negatives) < self.neg_sampling_ratio // 2 and attempts < 100:
            candidate = random.choice(all_listings)
            if candidate != central_listing and candidate not in positive_contexts:
                random_negatives.append(candidate)
            attempts += 1

        negatives.extend(random_negatives)

        # Hard negatives: same neighborhood but not in context
        if central_city and central_city in neighborhood_index:
            same_city_listings = neighborhood_index[central_city]
            hard_negatives = []
            attempts = 0

            while (
                len(hard_negatives) < self.neg_sampling_ratio - len(random_negatives)
                and attempts < 100
            ):
                candidate = random.choice(same_city_listings)
                if (
                    candidate != central_listing
                    and candidate not in positive_contexts
                    and candidate not in negatives
                ):
                    hard_negatives.append(candidate)
                attempts += 1

            negatives.extend(hard_negatives)

        # Fill remaining slots with random negatives if needed
        while len(negatives) < self.neg_sampling_ratio:
            candidate = random.choice(all_listings)
            if (
                candidate != central_listing
                and candidate not in positive_contexts
                and candidate not in negatives
            ):
                negatives.append(candidate)

        return negatives[: self.neg_sampling_ratio]

    def generate_triplets(
        self, sessions: List[Dict], listings: List[Dict]
    ) -> List[Dict[str, Any]]:
        """Generate training triplets from sessions."""
        # Build helper indexes
        all_listing_ids = [listing["id"] for listing in listings]
        neighborhood_index = self.build_neighborhood_index(listings)
        listing_to_city = {listing["id"]: listing["city"] for listing in listings}

        triplets = []

        print(f"Processing {len(sessions)} sessions...")

        for session_idx, session in enumerate(sessions):
            if session_idx % 100 == 0:
                print(f"  Processed {session_idx} sessions...")

            # Extract positive pairs from session
            positive_pairs = self.extract_sliding_window_pairs(session)

            # Group positive pairs by central listing
            central_to_contexts = defaultdict(set)
            for central, context in positive_pairs:
                central_to_contexts[central].add(context)

            # Generate triplets for each central listing
            for central_listing, positive_contexts in central_to_contexts.items():
                # Add positive triplets
                for context_listing in positive_contexts:
                    triplets.append(
                        {
                            "central_listing": central_listing,
                            "context_listing": context_listing,
                            "label": 1,
                            "session_id": session["session_id"],
                            "triplet_type": "positive",
                        }
                    )

                # Generate negative samples
                negative_listings = self.generate_negative_samples(
                    central_listing,
                    positive_contexts,
                    all_listing_ids,
                    neighborhood_index,
                    listing_to_city,
                )

                # Add negative triplets
                for neg_listing in negative_listings:
                    triplet_type = (
                        "hard_negative"
                        if listing_to_city.get(neg_listing)
                        == listing_to_city.get(central_listing)
                        else "random_negative"
                    )

                    triplets.append(
                        {
                            "central_listing": central_listing,
                            "context_listing": neg_listing,
                            "label": 0,
                            "session_id": session["session_id"],
                            "triplet_type": triplet_type,
                        }
                    )

        print(f"Generated {len(triplets)} triplets")
        return triplets
