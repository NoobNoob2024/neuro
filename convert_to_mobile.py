import torch
import torch.jit
import torchaudio
from model import MultiModalModel, ImageEncoder, SpeechEncoder

# --- Wrapper Models for Single-Modality Export ---

class ImageModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.image_encoder = ImageEncoder()
        self.image_encoder.load_state_dict(model.image_encoder.state_dict())
        self.image_classifier = torch.nn.Linear(128, 10)
        self.image_classifier.load_state_dict(model.image_classifier.state_dict())

    def forward(self, x_image: torch.Tensor) -> torch.Tensor:
        img_features = self.image_encoder(x_image)
        output = self.image_classifier(img_features)
        return output

class SpeechModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.transform = torch.nn.Sequential(
            torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=1024, n_mels=128)
        )
        self.speech_encoder = SpeechEncoder()
        self.speech_encoder.load_state_dict(model.speech_encoder.state_dict())
        self.speech_classifier = torch.nn.Linear(128, 10)
        self.speech_classifier.load_state_dict(model.speech_classifier.state_dict())
        
    def forward(self, x_speech: torch.Tensor) -> torch.Tensor:
        x_spec = self.transform(x_speech)
        # Add a batch dimension for the Conv2d layers
        x_spec = x_spec.unsqueeze(0)
        sp_features = self.speech_encoder(x_spec)
        output = self.speech_classifier(sp_features)
        return output

def main():
    # --- 1. Load the trained FP32 model ---
    device = torch.device("cpu")
    full_model = MultiModalModel()
    full_model.load_state_dict(torch.load("multimodal_model.pth", map_location=device))
    full_model.eval()
    print("Successfully loaded trained FP32 model.")

    # --- 2. Create and Trace the Image model ---
    print("\nProcessing Image model...")
    image_model_wrapper = ImageModelWrapper(full_model).eval()
    dummy_image = torch.randn(1, 1, 28, 28)
    traced_image_model = torch.jit.trace(image_model_wrapper, dummy_image)
    
    image_model_path = "image_model_mobile.ptl"
    traced_image_model._save_for_lite_interpreter(image_model_path)
    print(f"Image model saved to {image_model_path}")

    # --- 3. Create and Trace the Speech model ---
    print("\nProcessing Speech model...")
    speech_model_wrapper = SpeechModelWrapper(full_model).eval()
    dummy_speech_waveform = torch.randn(1, 16000)
    traced_speech_model = torch.jit.trace(speech_model_wrapper, dummy_speech_waveform)

    speech_model_path = "speech_model_mobile.ptl"
    traced_speech_model._save_for_lite_interpreter(speech_model_path)
    print(f"Speech model saved to {speech_model_path}")

    # --- 4. Verification ---
    print("\n--- Verifying converted models ---")
    
    loaded_image_model = torch.jit.load(image_model_path)
    output = loaded_image_model(dummy_image)
    print(f"Verified Image model output shape: {output.shape}")

    loaded_speech_model = torch.jit.load(speech_model_path)
    output = loaded_speech_model(dummy_speech_waveform)
    print(f"Verified Speech model output shape: {output.shape}")
    
    print("\nConversion to PyTorch Mobile format is complete and verified.")


if __name__ == "__main__":
    main()
