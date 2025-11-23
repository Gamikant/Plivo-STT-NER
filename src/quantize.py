import os
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

def main():
    model_dir = "out"
    out_dir = "out_quantized"
    
    print(f"Loading model from {model_dir}...")
    # Load the trained model
    model = AutoModelForTokenClassification.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    
    # Set to eval mode
    model.eval()

    print("Applying Dynamic Quantization...")
    # Quantize the Linear layers to int8
    quantized_model = torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )
    
    # Create a wrapper to ensure clean input/output for TorchScript
    class TracedWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
            
        def forward(self, input_ids, attention_mask):
            # We only care about logits for inference
            out = self.model(input_ids=input_ids, attention_mask=attention_mask)
            return out.logits

    wrapper = TracedWrapper(quantized_model)

    # Create dummy input for tracing
    print("Tracing model with TorchScript...")
    dummy_text = "This is a simple dummy sentence to trace the model."
    inputs = tokenizer(dummy_text, return_tensors="pt")
    
    # Trace the model
    traced_model = torch.jit.trace(
        wrapper, 
        (inputs["input_ids"], inputs["attention_mask"])
    )
    
    # Save
    os.makedirs(out_dir, exist_ok=True)
    model_path = os.path.join(out_dir, "model.pt")
    torch.jit.save(traced_model, model_path)
    tokenizer.save_pretrained(out_dir)
    
    print(f"Quantized and traced model saved to: {model_path}")
    print("You can now use this model for ultra-fast inference.")

if __name__ == "__main__":
    main()