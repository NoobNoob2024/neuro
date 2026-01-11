
import torch
import torch.quantization
import os
import torchaudio
from model import MultiModalModel
from dataset import get_speech_commands_loaders

def print_model_size(model, label):
    """Prints the size of the model's state_dict."""
    torch.save(model.state_dict(), "temp_model.p")
    size = os.path.getsize("temp_model.p") / 1e6  # in MB
    print(f"Size of {label}: {size:.2f} MB")
    os.remove("temp_model.p")

# Copied from train.py and modified to accept the transform as an argument
def test_epoch(model, data_loader, loss_fn, device, modality, mel_transform):
    model.eval()
    test_loss = 0
    correct = 0
    
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)

            if modality == "speech":
                data = mel_transform(data)
            
            if modality == "image":
                output = model(x_image=data)
            elif modality == "speech":
                output = model(x_speech=data)

            test_loss += loss_fn(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(data_loader.dataset)
    accuracy = 100. * correct / len(data_loader.dataset)
    print(f'\n{modality.capitalize()} Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(data_loader.dataset)} ({accuracy:.2f}%)\n')
    return accuracy


def main():
    # --- 1. Setup ---
    # Note: Quantization is done on the CPU
    device = torch.device("cpu")
    
    # Define a local mel_spectrogram_transform on the CPU
    mel_spectrogram_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=16000,
        n_fft=1024,
        n_mels=128
    ).to(device)
    
    # Load two copies of the model: one for baseline, one to be quantized.
    original_model = MultiModalModel()
    original_model.load_state_dict(torch.load("multimodal_model.pth", map_location=device))
    original_model.eval()
    
    model_to_quantize = MultiModalModel()
    model_to_quantize.load_state_dict(torch.load("multimodal_model.pth", map_location=device))
    model_to_quantize.eval()

    # --- 2. Fuse Modules ---
    # We will quantize the speech path. Let's fuse its modules first on the copy.
    # The layers to fuse are in the 'features' sequential block of the SpeechEncoder.
    # The pattern is (Conv, BN, ReLU)
    torch.quantization.fuse_modules(model_to_quantize.speech_encoder.features, [
        ['0', '1', '2'],
        ['4', '5', '6'],
        ['8', '9', '10'],
    ], inplace=True)
    
    # --- 3. Prepare for Static Quantization ---
    # Set the backend engine for quantized operations
    torch.backends.quantized.engine = 'qnnpack'
    
    # Specify quantization configuration for mobile ('qnnpack')
    model_to_quantize.qconfig = torch.quantization.get_default_qconfig('qnnpack')
    
    # Prepare the model for static quantization. This inserts observers.
    print("Preparing model for static quantization...")
    torch.quantization.prepare(model_to_quantize, inplace=True)

    # --- 4. Calibrate the model ---
    # We need to run a few batches of data through the model to let the observers
    # collect statistics on the range of activations.
    print("Calibrating model with speech data...")
    _, test_loader = get_speech_commands_loaders() # Use test loader for calibration
    
    # We only need to calibrate the speech path
    count = 0
    for data, target in test_loader:
        if count >= 10: # Use ~10 batches for calibration
            break
        
        # The model expects data on the CPU for calibration
        spec = mel_spectrogram_transform(data)
        model_to_quantize(x_speech=spec)
        count += 1
    
    print("Calibration complete.")

    # --- 5. Convert to Quantized Model ---
    print("Converting to quantized model...")
    quantized_model = model_to_quantize # for clarity
    torch.quantization.convert(quantized_model, inplace=True)
    print("Conversion complete.")

    # --- 6. Compare and Verify ---
    print("\n--- Model Size Comparison ---")
    print_model_size(original_model, "Original FP32 Model")
    print_model_size(quantized_model, "Quantized INT8 Model")

    print("\n--- Accuracy Comparison ---")
    loss_fn = torch.nn.CrossEntropyLoss()

    print("Evaluating original model accuracy:")
    test_epoch(original_model, test_loader, loss_fn, device, "speech", mel_spectrogram_transform)

    print("Evaluating quantized model accuracy:")
    test_epoch(quantized_model, test_loader, loss_fn, device, "speech", mel_spectrogram_transform)
    
    # Save the quantized model
    quantized_model_path = "multimodal_model_quantized.pth"
    torch.save(quantized_model.state_dict(), quantized_model_path)
    print(f"\nQuantized model saved to {quantized_model_path}")

if __name__ == "__main__":
    main()
