import torch
import torchvision
import torchaudio
import os

def download_datasets():
    """
    Downloads the MNIST and SpeechCommands datasets.
    """
    # Create a directory to store the data
    data_dir = 'data'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    print("Downloading MNIST dataset...")
    try:
        train_set_mnist = torchvision.datasets.MNIST(
            root=data_dir, train=True, download=True
        )
        test_set_mnist = torchvision.datasets.MNIST(
            root=data_dir, train=False, download=True
        )
        print("MNIST dataset downloaded successfully.")
    except Exception as e:
        print(f"Error downloading MNIST: {e}")


    print("\nDownloading SpeechCommands dataset...")
    try:
        speech_commands_dataset = torchaudio.datasets.SPEECHCOMMANDS(
            root=data_dir, download=True
        )
        print("SpeechCommands dataset downloaded successfully.")
    except Exception as e:
        print(f"Error downloading SpeechCommands: {e}")


if __name__ == '__main__':
    download_datasets()
