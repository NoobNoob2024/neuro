
import torch
import torch.jit
from tqdm import tqdm

from dataset import get_mnist_loaders, get_speech_commands_loaders

def test_mobile_model(model, data_loader, device, modality):
    """
    A custom test loop for scripted mobile models.
    """
    model.to(device)
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        progress_bar = tqdm(data_loader, desc=f"Verifying {modality} model")
        for data, target in progress_bar:
            # For scripted speech model, we don't need to apply transform,
            # as it's part of the model. But we do need to handle the batch dimension.
            # The dataloader gives a batch, but our model expects a single item.
            # So we iterate through the batch.
            
            if modality == "image":
                data, target = data.to(device), target.to(device)
                output = model(data)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += data.size(0)
            elif modality == "speech":
                # The speech model expects a single waveform, so we loop through the batch
                data, target = data.to(device), target.to(device)
                for i in range(data.size(0)):
                    single_waveform = data[i]
                    output = model(single_waveform) # Pass single waveform
                    pred = output.argmax(dim=1, keepdim=True)
                    correct += pred.eq(target[i].view_as(pred)).sum().item()
                    total += 1

    accuracy = 100. * correct / total
    print(f'\n{modality.capitalize()} Mobile Model Accuracy: {correct}/{total} ({accuracy:.2f}%)\n')
    return accuracy

def main():
    device = torch.device("cpu") # Mobile models run on CPU
    print(f"Using device: {device}")

    # --- 1. Verify Image Model ---
    print("--- Verifying Image Mobile Model ---")
    image_model = torch.jit.load("image_model_mobile.ptl", map_location=device)
    _, test_loader_mnist = get_mnist_loaders()
    test_mobile_model(image_model, test_loader_mnist, device, "image")

    # --- 2. Verify Speech Model ---
    print("\n--- Verifying Speech Mobile Model ---")
    speech_model = torch.jit.load("speech_model_mobile.ptl", map_location=device)
    _, test_loader_speech = get_speech_commands_loaders()
    test_mobile_model(speech_model, test_loader_speech, device, "speech")

if __name__ == "__main__":
    main()
