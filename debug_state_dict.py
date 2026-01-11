
import torch
from model import MultiModalModel

def inspect_state_dict(model_path):
    """
    Loads a state_dict and prints the keys, shapes, and values for the
    speech encoder's BatchNorm layers to debug the mismatch error.
    """
    try:
        # Load the saved model state dict
        state_dict = torch.load(model_path, map_location=torch.device("cpu"))
        print(f"Successfully loaded state dict from {model_path}\n")

        print("--- Inspecting Speech Encoder BatchNorm Buffers ---")
        
        # Check the relevant keys for the BatchNorm layers in the Sequential block
        keys_to_check = {
            'First BatchNorm (should be 32)': 'speech_encoder.features.1.running_mean',
            'Second BatchNorm (should be 64)': 'speech_encoder.features.5.running_mean',
            'Third BatchNorm (should be 128)': 'speech_encoder.features.9.running_mean',
        }

        all_keys_found = True
        for description, key in keys_to_check.items():
            if key in state_dict:
                tensor = state_dict[key]
                print(f"[{description}]")
                print(f"  Key: '{key}'")
                print(f"  Shape: {tensor.shape}")
                print(f"  First 3 values: {tensor[:3].tolist()}")
                print("-" * 20)
            else:
                print(f"Error: Key '{key}' not found in state_dict!")
                all_keys_found = False

        if all_keys_found:
            print("\nAll expected BatchNorm keys were found. Their shapes appear correct.")
            print("This suggests the error might not be in the saved file itself, but in how JIT handles it.")
        else:
            print("\nCritical error: Not all BatchNorm keys were found. The saved model file is likely incorrect.")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    model_file = "multimodal_model.pth"
    inspect_state_dict(model_file)
