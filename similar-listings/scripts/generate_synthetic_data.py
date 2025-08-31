#!/usr/bin/env python3
"""Generate synthetic data for similar listings recommendation system."""

import json
import uuid
import random
from datetime import datetime, timedelta
from typing import List, Dict, Any
import argparse


class SyntheticDataGenerator:
    def __init__(self, seed: int = 42):
        random.seed(seed)
        self.cities = [
            ("New York", "US"),
            ("Los Angeles", "US"),
            ("Chicago", "US"),
            ("Toronto", "CA"),
            ("Vancouver", "CA"),
            ("London", "UK"),
            ("Paris", "FR"),
            ("Berlin", "DE"),
            ("Tokyo", "JP"),
            ("Sydney", "AU"),
        ]
        self.listing_types = [
            "apartment",
            "house",
            "condo",
            "loft",
            "studio",
            "townhouse",
        ]
        self.genders = ["male", "female", "other", "prefer_not_to_say"]
        self.languages = ["en", "fr", "de", "es", "ja", "pt", "it", "zh", "ko"]
        self.timezones = [
            "America/New_York",
            "America/Los_Angeles",
            "America/Chicago",
            "America/Toronto",
            "Europe/London",
            "Europe/Paris",
            "Europe/Berlin",
            "Asia/Tokyo",
            "Australia/Sydney",
        ]
        self.interaction_types = ["view", "click", "favorite", "book", "inquire"]
        self.sources = ["search", "featured", "recommendation", "direct", "social"]

        # Base timestamp (30 days ago)
        self.base_timestamp = int((datetime.now() - timedelta(days=30)).timestamp())

    def generate_users(self, num_users: int) -> List[Dict[str, Any]]:
        """Generate synthetic users."""
        users = []
        for i in range(1, num_users + 1):
            city, country = random.choice(self.cities)
            user = {
                "id": i,
                "username": f"user_{i:06d}",
                "age": random.randint(18, 75),
                "gender": random.choice(self.genders),
                "city": city,
                "country": country,
                "language": random.choice(self.languages),
                "time_zone": random.choice(self.timezones),
            }
            users.append(user)
        return users

    def generate_listings(
        self, num_listings: int, user_ids: List[int]
    ) -> List[Dict[str, Any]]:
        """Generate synthetic listings."""
        listings = []
        for _ in range(num_listings):
            city, _ = random.choice(self.cities)
            listing_type = random.choice(self.listing_types)
            beds = random.randint(0, 5)

            # Price varies by city and type
            base_price = random.uniform(50, 500)
            if city in ["New York", "Los Angeles", "London", "Paris", "Tokyo"]:
                base_price *= random.uniform(1.5, 3.0)

            listing = {
                "id": str(uuid.uuid4()),
                "host_id": random.choice(user_ids),
                "price": round(base_price, 2),
                "sq_ft": random.randint(300, 3000),
                "rate": round(random.uniform(3.5, 5.0), 1),
                "type": listing_type,
                "city": city,
                "beds": beds,
                "max_guests": random.randint(1, min(8, beds * 2 + 2)),
            }
            listings.append(listing)
        return listings

    def generate_interactions(
        self, num_interactions: int, user_ids: List[int], listing_ids: List[str]
    ) -> List[Dict[str, Any]]:
        """Generate synthetic user-listing interactions with realistic browsing patterns."""
        interactions = []

        # Generate sessions where users browse multiple listings
        num_sessions = num_interactions // 5  # Average 5 interactions per session

        for _ in range(num_sessions):
            user_id = random.choice(user_ids)
            session_timestamp = self.base_timestamp + random.randint(0, 30 * 24 * 3600)

            # Generate a browsing session with 2-8 interactions
            session_length = random.randint(2, 8)
            session_listings = random.sample(
                listing_ids, min(session_length, len(listing_ids))
            )

            for position, listing_id in enumerate(session_listings):
                interaction = {
                    "id": str(uuid.uuid4()),
                    "user_id": user_id,
                    "listing_id": listing_id,
                    "listing_display_position": position + 1,
                    "interaction_type": self._choose_interaction_type(position),
                    "source": random.choice(self.sources),
                    "timestamp": session_timestamp
                    + random.randint(0, 3600),  # Within 1 hour
                }
                interactions.append(interaction)

        # Fill remaining interactions if needed
        while len(interactions) < num_interactions:
            interaction = {
                "id": str(uuid.uuid4()),
                "user_id": random.choice(user_ids),
                "listing_id": random.choice(listing_ids),
                "listing_display_position": random.randint(1, 20),
                "interaction_type": random.choice(self.interaction_types),
                "source": random.choice(self.sources),
                "timestamp": self.base_timestamp + random.randint(0, 30 * 24 * 3600),
            }
            interactions.append(interaction)

        return interactions[:num_interactions]

    def _choose_interaction_type(self, position: int) -> str:
        """Choose interaction type based on position in session (earlier = more likely to be view)."""
        if position == 0:
            return random.choices(
                self.interaction_types,
                weights=[0.7, 0.15, 0.05, 0.05, 0.05],  # Mostly views at start
                k=1,
            )[0]
        else:
            return random.choices(
                self.interaction_types,
                weights=[0.4, 0.3, 0.1, 0.1, 0.1],  # More diverse later
                k=1,
            )[0]

    def generate_all(
        self,
        num_users: int = 1000,
        num_listings: int = 5000,
        num_interactions: int = 50000,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Generate all synthetic datasets."""
        print(f"Generating {num_users} users...")
        users = self.generate_users(num_users)
        user_ids = [user["id"] for user in users]

        print(f"Generating {num_listings} listings...")
        listings = self.generate_listings(num_listings, user_ids)
        listing_ids = [listing["id"] for listing in listings]

        print(f"Generating {num_interactions} interactions...")
        interactions = self.generate_interactions(
            num_interactions, user_ids, listing_ids
        )

        return {"users": users, "listings": listings, "interactions": interactions}

    def save_to_files(
        self, data: Dict[str, List[Dict[str, Any]]], output_dir: str = "data"
    ):
        """Save datasets to JSON files."""
        import os

        os.makedirs(output_dir, exist_ok=True)

        for dataset_name, dataset in data.items():
            filepath = os.path.join(output_dir, f"{dataset_name}.json")
            print(f"Saving {len(dataset)} {dataset_name} to {filepath}")
            with open(filepath, "w") as f:
                json.dump(dataset, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic data for recommendation system"
    )
    parser.add_argument(
        "--users", type=int, default=1000, help="Number of users to generate"
    )
    parser.add_argument(
        "--listings", type=int, default=5000, help="Number of listings to generate"
    )
    parser.add_argument(
        "--interactions",
        type=int,
        default=50000,
        help="Number of interactions to generate",
    )
    parser.add_argument(
        "--output-dir", type=str, default="data", help="Output directory for JSON files"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )

    args = parser.parse_args()

    generator = SyntheticDataGenerator(seed=args.seed)
    data = generator.generate_all(args.users, args.listings, args.interactions)
    generator.save_to_files(data, args.output_dir)

    print("\nSynthetic data generation completed!")
    print(f"Users: {len(data['users'])}")
    print(f"Listings: {len(data['listings'])}")
    print(f"Interactions: {len(data['interactions'])}")


if __name__ == "__main__":
    main()
