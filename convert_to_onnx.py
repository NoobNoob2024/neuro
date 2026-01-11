
import torch
import torch.onnx
from model import MultiModalModel, ImageEncoder, SpeechEncoder

# This wrapper is identical to the one from the mobile conversion.
# It creates a self-contained model for image recognition.
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

def main():
    # --- 1. Load the trained FP32 model ---
    device = torch.device("cpu")
    full_model = MultiModalModel()
    full_model.load_state_dict(torch.load("multimodal_model.pth", map_location=device))
    full_model.eval()
    print("Successfully loaded trained FP32 model.")

    # --- 2. Create the wrapper model for ONNX export ---
    image_model_wrapper = ImageModelWrapper(full_model).eval()
    print("Created image model wrapper for ONNX export.")

    # --- 3. Export to ONNX ---
    # Create a dummy input tensor with the correct shape
    # N, C, H, W -> Batch Size, Channels, Height, Width
    dummy_input = torch.randn(1, 1, 28, 28, requires_grad=True)
    
    output_onnx_file = "image_model.onnx"

    print(f"Exporting model to {output_onnx_file}...")
    
    torch.onnx.export(
        image_model_wrapper,        # The model to export
        dummy_input,                # Model input (or a tuple for multiple inputs)
        output_onnx_file,           # Where to save the model
        export_params=True,         # Store the trained parameter weights inside the model file
        opset_version=11,           # The ONNX version to export the model to
        do_constant_folding=True,   # Whether to execute constant folding for optimization
        input_names=['input'],      # The model's input names
        output_names=['output'],    # The model's output names
        dynamic_axes={'input' : {0 : 'batch_size'},    # Variable length axes
                      'output' : {0 : 'batch_size'}}
    )

    print(f"Model has been converted to ONNX format and saved as {output_onnx_file}")
    print("You can now use this file with ONNX Runtime in a web browser.")


if __name__ == "__main__":
    main()
