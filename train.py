import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os

from model import MultiModalModel
from dataset import get_mnist_loaders, get_speech_commands_loaders
import torchaudio


# --- Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Move the Mel spectrogram transform to the correct device to avoid runtime errors
mel_spectrogram_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=16000,
    n_fft=1024,
    n_mels=128
).to(DEVICE)
LEARNING_RATE = 0.001
NUM_EPOCHS_MNIST = 3
NUM_EPOCHS_SPEECH = 10 # Speech is a bit harder, so more epochs
MODEL_SAVE_PATH = "multimodal_model.pth"

# --- Training and Evaluation Functions ---

def train_epoch(model, data_loader, loss_fn, optimizer, device, modality):
    model.train()
    total_loss = 0.0
    
    progress_bar = tqdm(data_loader, desc=f"Training {modality}")
    for batch_idx, (data, target) in enumerate(progress_bar):
        data, target = data.to(device), target.to(device)

        # For speech, apply the spectrogram transform
        if modality == "speech":
            data = mel_spectrogram_transform(data)

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        if modality == "image":
            output = model(x_image=data)
        elif modality == "speech":
            output = model(x_speech=data)
        
        loss = loss_fn(output, target)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        progress_bar.set_postfix(loss=total_loss/(batch_idx+1))

def test_epoch(model, data_loader, loss_fn, device, modality):
    model.eval()
    test_loss = 0
    correct = 0
    
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)

            if modality == "speech":
                data = mel_spectrogram_transform(data)
            
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

# --- Main Execution ---

if __name__ == "__main__":
    print(f"Using device: {DEVICE}")

    # 1. Load Data
    train_loader_mnist, test_loader_mnist = get_mnist_loaders()
    train_loader_speech, test_loader_speech = get_speech_commands_loaders()

    # 2. Initialize Model, Loss, and Optimizer
    model = MultiModalModel().to(DEVICE)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 3. Train on MNIST
    print("--- Starting MNIST Training ---")
    for epoch in range(1, NUM_EPOCHS_MNIST + 1):
        print(f"Epoch {epoch}/{NUM_EPOCHS_MNIST}")
        train_epoch(model, train_loader_mnist, loss_fn, optimizer, DEVICE, "image")
        test_epoch(model, test_loader_mnist, loss_fn, DEVICE, "image")

    # 4. Train on SpeechCommands
    print("\n--- Starting SpeechCommands Training ---")
    for epoch in range(1, NUM_EPOCHS_SPEECH + 1):
        print(f"Epoch {epoch}/{NUM_EPOCHS_SPEECH}")
        train_epoch(model, train_loader_speech, loss_fn, optimizer, DEVICE, "speech")
        test_epoch(model, test_loader_speech, loss_fn, DEVICE, "speech")

    # 5. Save the trained model
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"\nTraining complete. Model saved to {MODEL_SAVE_PATH}")
