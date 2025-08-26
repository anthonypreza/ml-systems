import torch
from torch.utils.data import Dataset

import os
import glob
import random
import subprocess
from typing import List, Dict, Optional
from tqdm import tqdm

from transformers import AutoTokenizer
from datasets import load_dataset

from .video import VideoProcessor


class VideoQueryTriplet:
    """Data structure for training triplets: (video_path, positive_query, negative_queries)"""

    def __init__(
        self, video_path: str, positive_query: str, negative_queries: List[str]
    ):
        self.video_path = video_path
        self.positive_query = positive_query
        self.negative_queries = negative_queries

    def get_all_queries(self) -> List[str]:
        """Returns [positive_query, negative_query1, negative_query2, ...]"""
        return [self.positive_query] + self.negative_queries

    def get_target_index(self) -> int:
        """Returns the index of the positive query (always 0)"""
        return 0


class VideoQueryDataset(Dataset):
    """Dataset class for loading and processing video-query triplets"""

    def __init__(
        self,
        triplets: List[VideoQueryTriplet],
        video_processor: VideoProcessor,
        tokenizer: AutoTokenizer,
        max_text_length: int = 128,
    ):
        self.triplets = triplets
        self.video_processor = video_processor
        self.tokenizer = tokenizer
        self.max_text_length = max_text_length

    def __len__(self) -> int:
        return len(self.triplets)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        triplet = self.triplets[idx]

        # Extract video frames
        video_frames = self.video_processor.extract_frames(triplet.video_path)

        # Get all queries (positive + negatives)
        all_queries = triplet.get_all_queries()
        target_index = triplet.get_target_index()

        # Tokenize all queries
        encoding = self.tokenizer(
            all_queries,
            truncation=True,
            padding="max_length",
            max_length=self.max_text_length,
            return_tensors="pt",
        )  # type: ignore

        return {
            "video_frames": video_frames,  # [num_frames, 3, H, W]
            "text_input_ids": encoding["input_ids"],  # [num_texts, seq_len]
            "text_attention_mask": encoding["attention_mask"],  # [num_texts, seq_len]
            "target_index": torch.tensor(target_index, dtype=torch.long),  # scalar
            "video_path": triplet.video_path,  # for debugging
            "queries": all_queries,  # for debugging
        }  # type: ignore


def download_youtube_clip(
    youtube_url: str, output_path: str, start_time: float, end_time: float
) -> bool:
    """Download a specific clip from a YouTube video using yt-dlp and ffmpeg"""
    try:
        print(
            f"Downloading video from {youtube_url}, start_time={start_time} and end_time={end_time}"
        )
        # Create temporary file for full video
        temp_video = output_path + ".temp.%(ext)s"
        duration = end_time - start_time

        # Download the specific segment using yt-dlp with ffmpeg
        result = subprocess.run(
            [
                "yt-dlp",
                "--external-downloader",
                "ffmpeg",
                "--external-downloader-args",
                f"ffmpeg_i:-ss {start_time} -t {duration}",
                "-o",
                temp_video,
                "--no-playlist",
                "--format",
                "best[height<=480]",  # Lower quality for faster download
                youtube_url,
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )

        if result.returncode != 0:
            print(f"yt-dlp failed: {result.stderr}")
            return False

        # Find the downloaded file (yt-dlp adds extension)
        temp_files = glob.glob(output_path + ".temp.*")
        if not temp_files:
            print(f"No temporary file found for {youtube_url}")
            return False

        # Move to final location
        temp_file = temp_files[0]
        os.rename(temp_file, output_path)

        return os.path.exists(output_path) and os.path.getsize(output_path) > 0

    except Exception as e:
        print(f"Failed to download clip from {youtube_url}: {e}")
        # Clean up any temporary files
        for temp_file in glob.glob(output_path + ".temp.*"):
            try:
                os.remove(temp_file)
            except:
                pass
        return False


def load_msrvtt_dataset(
    data_dir: str = "./data/msrvtt",
    max_videos: Optional[int] = None,
    num_negatives: int = 7,
) -> List[VideoQueryTriplet]:
    """Load MSR-VTT dataset and convert to VideoQueryTriplet format

    Args:
        data_dir: Directory to store downloaded videos
        max_videos: Maximum number of videos to process (None for all)
        num_negatives: Number of negative captions per video

    Returns:
        List of VideoQueryTriplet objects
    """
    print("Loading MSR-VTT dataset...")
    dataset = load_dataset("friedrichor/MSR-VTT", "train_7k")

    # Use train split
    train_data = dataset["train"]
    if max_videos:
        train_data = train_data.select(range(min(max_videos, len(train_data))))  # type: ignore

    # Create video directory
    video_dir = os.path.join(data_dir, "videos")
    os.makedirs(video_dir, exist_ok=True)

    # Group captions by video_id and collect all captions
    video_captions = {}
    all_captions = []

    for i, item in enumerate(tqdm(train_data, desc="Processing captions")):
        video_id = item["video_id"]
        caption = item["caption"]
        video_url = item["url"]
        start_time = item.get("start time", 0)  # Default to 0 if not available
        end_time = item.get("end time", 10)  # Default to 10 seconds if not available

        # Handle case where caption might be a list
        if isinstance(caption, list):
            captions_to_add = caption
        else:
            captions_to_add = [caption]

        # Add to all captions for negative sampling
        for cap in captions_to_add:
            all_captions.append(cap)

        if video_id not in video_captions:
            video_captions[video_id] = {
                "captions": [],
                "url": video_url,
                "start_time": start_time,
                "end_time": end_time,
            }

        for cap in captions_to_add:
            video_captions[video_id]["captions"].append(cap)

    triplets = []
    failed_downloads = 0

    for video_id, video_data in tqdm(video_captions.items(), desc="Creating triplets"):
        video_path = os.path.join(video_dir, f"{video_id}.mp4")

        # Download video clip if not exists
        if not os.path.exists(video_path):
            if not download_youtube_clip(
                video_data["url"],
                video_path,
                video_data["start_time"],
                video_data["end_time"],
            ):
                failed_downloads += 1
                continue

        # For each caption, create a separate triplet
        for caption in video_data["captions"]:
            # Sample negative captions (from other videos, excluding this caption)
            other_captions = [c for c in all_captions if c != caption]
            negative_queries = random.sample(
                other_captions, min(num_negatives, len(other_captions))
            )

            triplet = VideoQueryTriplet(
                video_path=video_path,
                positive_query=caption,  # Single caption, not a list
                negative_queries=negative_queries,  # List of negative captions
            )
            triplets.append(triplet)

    print(f"Created {len(triplets)} triplets from {len(video_captions)} videos")
    if failed_downloads > 0:
        print(f"Failed to download {failed_downloads} videos")

    return triplets


def create_synthetic_dataset(
    num_videos: int = 20, num_negatives: int = 7
) -> List[VideoQueryTriplet]:
    """Create a synthetic dataset for testing"""

    # Create some dummy videos
    video_dir = "/tmp/video_search_test"
    os.makedirs(video_dir, exist_ok=True)

    # Video categories and their associated queries
    categories = {
        "cats": {
            "positive_queries": [
                "cat playing with toy",
                "kitten running around",
                "cat sleeping peacefully",
                "cats chasing laser pointer",
                "cute cat meowing",
                "tabby cat eating",
            ],
            "video_style": "cat_video",
        },
        "dogs": {
            "positive_queries": [
                "dog running in park",
                "puppy playing fetch",
                "golden retriever swimming",
                "dog barking at stranger",
                "cute puppy sleeping",
                "dog catching frisbee",
            ],
            "video_style": "dog_video",
        },
        "cooking": {
            "positive_queries": [
                "cooking pasta in kitchen",
                "chopping vegetables",
                "baking chocolate cake",
                "frying eggs in pan",
                "making pizza dough",
                "grilling chicken",
            ],
            "video_style": "cooking_video",
        },
        "sports": {
            "positive_queries": [
                "basketball player shooting",
                "soccer match goal",
                "tennis player serving",
                "running marathon race",
                "swimming competition",
                "football touchdown",
            ],
            "video_style": "sports_video",
        },
    }

    # All possible queries for negatives
    all_queries = []
    for cat_data in categories.values():
        all_queries.extend(cat_data["positive_queries"])

    triplets = []
    video_processor = VideoProcessor()

    for i in range(num_videos):
        # Choose random category
        category = random.choice(list(categories.keys()))
        cat_data = categories[category]

        # Create video file
        video_path = os.path.join(video_dir, f"{category}_{i}.mp4")
        if not os.path.exists(video_path):
            video_processor.create_dummy_video(video_path, duration_seconds=3)

        # Choose positive query
        positive_query = random.choice(cat_data["positive_queries"])

        # Choose negative queries (from other categories or different queries from same category)
        other_queries = [q for q in all_queries if q != positive_query]
        negative_queries = random.sample(
            other_queries, min(num_negatives, len(other_queries))
        )

        # Create triplet
        triplet = VideoQueryTriplet(
            video_path=video_path,
            positive_query=positive_query,
            negative_queries=negative_queries,
        )

        triplets.append(triplet)

    return triplets
