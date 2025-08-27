import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from typing import Tuple


class VideoProcessor:
    """Handles video frame extraction and preprocessing"""

    def __init__(self, max_frames: int = 8, frame_size: Tuple[int, int] = (224, 224)):
        self.max_frames = max_frames
        self.frame_size = frame_size

        # Image preprocessing pipeline
        self.transform = transforms.Compose(
            [
                transforms.Resize(frame_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def extract_frames(self, video_path: str) -> torch.Tensor:
        """Extract frames from video and return as tensor [num_frames, 3, H, W]"""
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Cannot open video: {video_path}")

            # Get video info
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames == 0:
                raise ValueError(f"No frames found in video: {video_path}")

            # Sample frame indices uniformly
            if total_frames <= self.max_frames:
                frame_indices = list(range(total_frames))
            else:
                frame_indices = np.linspace(
                    0, total_frames - 1, self.max_frames, dtype=int
                )

            frames = []
            for frame_idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()

                if ret:
                    # Convert BGR to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    # Convert to PIL and apply transforms
                    frame_pil = Image.fromarray(frame_rgb)
                    frame_tensor = self.transform(frame_pil)
                    frames.append(frame_tensor)

            cap.release()

            if not frames:
                raise ValueError(f"No valid frames extracted from: {video_path}")

            # Stack frames and pad if necessary
            frames_tensor = torch.stack(frames)

            # Pad with zeros if we have fewer frames than max_frames
            if len(frames) < self.max_frames:
                padding = torch.zeros(
                    (self.max_frames - len(frames), 3, *self.frame_size)
                )
                frames_tensor = torch.cat([frames_tensor, padding], dim=0)

            return frames_tensor

        except Exception as e:
            print(f"Error processing video {video_path}: {e}")
            # Return dummy frames if processing fails
            return torch.zeros((self.max_frames, 3, *self.frame_size))

    def create_dummy_video(self, save_path: str, duration_seconds: int = 5):
        """Create a dummy video for testing purposes"""
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore
        out = cv2.VideoWriter(save_path, fourcc, 30.0, (640, 480))

        for i in range(duration_seconds * 30):  # 30 FPS
            # Create a simple colored frame
            color = (i % 256, (i * 2) % 256, (i * 3) % 256)
            frame = np.full((480, 640, 3), color, dtype=np.uint8)
            out.write(frame)

        out.release()
        print(f"Dummy video created at: {save_path}")
