import torch
import torch.nn as nn
import torch.nn.functional as F

class ImageEncoder(nn.Module):
    """Encodes MNIST images into a feature vector."""
    def __init__(self, feature_dim=128):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # MNIST images are 28x28, after 2 pooling layers, they become 7x7
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, feature_dim)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)  # Flatten the tensor
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class SpeechEncoder(nn.Module):
    """Encodes speech spectrograms into a feature vector."""
    def __init__(self, feature_dim=128):
        super().__init__()
        # Input spectrogram is (batch, 1, n_mels, time_steps)
        # e.g., (128, 1, 128, 32) from our dataset.py
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, feature_dim)

    def forward(self, x):
        x = self.features(x)
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1) # Flatten
        x = self.fc(x)
        return x

class MultiModalModel(nn.Module):
    """A multi-modal model for image and speech classification."""
    def __init__(self, num_classes=10, image_feature_dim=128, speech_feature_dim=128):
        super().__init__()
        self.image_encoder = ImageEncoder(feature_dim=image_feature_dim)
        self.speech_encoder = SpeechEncoder(feature_dim=speech_feature_dim)
        
        # Classifier that takes the concatenated feature vector
        self.classifier = nn.Linear(image_feature_dim + speech_feature_dim, num_classes)
        
        # We also need classifiers for single-modality inputs
        self.image_classifier = nn.Linear(image_feature_dim, num_classes)
        self.speech_classifier = nn.Linear(speech_feature_dim, num_classes)

    def forward(self, x_image=None, x_speech=None):
        if x_image is not None and x_speech is not None:
            # Multi-modal input
            img_features = self.image_encoder(x_image)
            sp_features = self.speech_encoder(x_speech)
            combined_features = torch.cat((img_features, sp_features), dim=1)
            output = self.classifier(combined_features)
        elif x_image is not None:
            # Image-only input
            img_features = self.image_encoder(x_image)
            output = self.image_classifier(img_features)
        elif x_speech is not None:
            # Speech-only input
            sp_features = self.speech_encoder(x_speech)
            output = self.speech_classifier(sp_features)
        else:
            raise ValueError("At least one input (x_image or x_speech) must be provided.")
            
        return output

if __name__ == '__main__':
    # --- Test the model with dummy inputs ---
    model = MultiModalModel()
    print(model)
    print("\n--- Testing Model Components ---")

    # Test image-only path
    dummy_image = torch.randn(16, 1, 28, 28) # Batch of 16 images
    image_output = model(x_image=dummy_image)
    print(f"Image-only output shape: {image_output.shape}") # Expected: [16, 10]

    # Test speech-only path
    dummy_speech_spec = torch.randn(16, 1, 128, 32) # Batch of 16 spectrograms
    speech_output = model(x_speech=dummy_speech_spec)
    print(f"Speech-only output shape: {speech_output.shape}") # Expected: [16, 10]

    # Test multi-modal path (not used for training, but for demonstration)
    mm_output = model(x_image=dummy_image, x_speech=dummy_speech_spec)
    print(f"Multi-modal output shape: {mm_output.shape}") # Expected: [16, 10]
    
    print("\nModel definition is complete and seems to work with dummy data.")
