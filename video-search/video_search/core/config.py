import torch


class VideoSearchConfig:
    """Configuration class for the video search model"""

    def __init__(self):
        # Video processing
        self.max_frames = 8
        self.frame_size = (224, 224)

        # Text processing
        self.max_text_length = 128
        self.text_model_name = "bert-base-uncased"

        # Model architecture
        self.video_embed_dim = 512
        self.text_embed_dim = 512
        self.temperature = 0.07

        # Training
        self.num_negatives = 7
        self.batch_size = 16
        self.learning_rate = 1e-4
        self.num_epochs = 10

        # Device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Save paths
        self.model_save_path = "./models/video_search_model.pth"
        self.index_save_path = "./models/video_index.pkl"
        self.video_paths_save_path = "./models/video_paths.txt"
