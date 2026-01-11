
import torch
import torchaudio
import torchvision
from torch.utils.data import DataLoader, Subset
import os

# --- Global Parameters ---
DATA_DIR = 'data'
BATCH_SIZE = 128
SAMPLE_RATE = 16000
N_FFT = 1024
N_MELS = 128
NUM_CLASSES = 10

# --- Label Mapping for SpeechCommands ---
LABELS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
label_to_index = {label: i for i, label in enumerate(LABELS)}

# --- Preprocessing Transforms ---
# For image data
mnist_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.1307,), (0.3081,))
])

# For audio data
mel_spectrogram_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=SAMPLE_RATE,
    n_fft=N_FFT,
    n_mels=N_MELS
)

# --- Collate Function for Speech Data ---
def pad_sequence(batch):
    # Make all tensors in a batch the same length by padding with zeros
    batch = [item.t() for item in batch]
    batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0.)
    return batch.permute(0, 2, 1)

def collate_fn_speech(batch):
    tensors, targets = [], []

    # Gather in lists, and encode labels as indices
    for waveform, sample_rate, label, speaker_id, utterance_number in batch:
        tensors += [waveform]
        targets += [label_to_index[label]]

    # Group the list of tensors into a batched tensor
    tensors = pad_sequence(tensors)
    targets = torch.tensor(targets, dtype=torch.long)

    return tensors, targets

# --- Data Loader Functions ---

def get_mnist_loaders():
    """Returns DataLoader for MNIST dataset."""
    train_dataset = torchvision.datasets.MNIST(
        root=DATA_DIR, train=True, download=False, transform=mnist_transform
    )
    test_dataset = torchvision.datasets.MNIST(
        root=DATA_DIR, train=False, download=False, transform=mnist_transform
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, test_loader

def get_speech_commands_loaders():
    """Returns DataLoader for SpeechCommands dataset, filtered for digits."""
    full_dataset = torchaudio.datasets.SPEECHCOMMANDS(root=DATA_DIR, download=False)

    # Filter out samples that are not in our LABELS list
    indices = []
    for i in range(len(full_dataset)):
        waveform, sample_rate, label, speaker_id, utterance_number = full_dataset[i]
        if label in LABELS:
            indices.append(i)
    
    filtered_dataset = Subset(full_dataset, indices)
    
    # Split into train and test
    # Note: A more robust split would use speaker IDs, but for simplicity we do a random split.
    num_train = int(len(filtered_dataset) * 0.8)
    num_test = len(filtered_dataset) - num_train
    train_set, test_set = torch.utils.data.random_split(filtered_dataset, [num_train, num_test])

    train_loader = DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn_speech
    )
    test_loader = DataLoader(
        test_set,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn_speech
    )

    return train_loader, test_loader

if __name__ == '__main__':
    print("Testing MNIST data loader...")
    train_loader_mnist, test_loader_mnist = get_mnist_loaders()
    img_batch, label_batch = next(iter(train_loader_mnist))
    print(f"Image batch shape: {img_batch.shape}")
    print(f"Label batch shape: {label_batch.shape}")
    print("-" * 20)

    print("Testing SpeechCommands data loader...")
    train_loader_speech, test_loader_speech = get_speech_commands_loaders()
    waveform_batch, label_batch_speech = next(iter(train_loader_speech))
    
    # Apply the transform here to see the output shape
    spectrograms = mel_spectrogram_transform(waveform_batch)
    
    print(f"Waveform batch shape: {waveform_batch.shape}")
    print(f"Spectrogram batch shape: {spectrograms.shape}")
    print(f"Speech label batch shape: {label_batch_speech.shape}")
    print("-" * 20)
    print("Data loaders are ready to be used.")
