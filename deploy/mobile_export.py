"""
Export transformer model to ONNX or TorchScript and quantize for mobile deployment.
"""
import argparse
import os
import torch
from transformers import AutoModelForSequenceClassification

def export_torchscript(model, output_path):
    model.eval()
    example = torch.randint(0, model.config.vocab_size, (1, 32))
    traced = torch.jit.trace(model, example)
    torch.jit.save(traced, output_path)
    print(f"Saved TorchScript model to {output_path}")

def export_onnx(model, output_path):
    model.eval()
    dummy_input = torch.randint(0, model.config.vocab_size, (1, 32))
    torch.onnx.export(model, dummy_input, output_path,
                      input_names=['input_ids'], output_names=['logits'],
                      dynamic_axes={'input_ids': {0: 'batch'}, 'logits': {0: 'batch'}},
                      opset_version=13)
    print(f"Saved ONNX model to {output_path}")

def quantize_dynamic(model, output_path):
    import torch.quantization
    quantized_model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
    torch.save(quantized_model.state_dict(), output_path)
    print(f"Saved quantized model weights to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Export model for mobile deployment.")
    parser.add_argument('--model_name', type=str, required=True, help='HuggingFace model name or path')
    parser.add_argument('--format', type=str, choices=['onnx', 'torchscript'], default='onnx')
    parser.add_argument('--quantize', action='store_true', help='Apply dynamic quantization')
    parser.add_argument('--output', type=str, required=True, help='Output file path')
    args = parser.parse_args()

    model = AutoModelForSequenceClassification.from_pretrained(args.model_name)
    if args.quantize:
        quantize_dynamic(model, args.output + '.quantized.pt')
    if args.format == 'onnx':
        export_onnx(model, args.output + '.onnx')
    elif args.format == 'torchscript':
        export_torchscript(model, args.output + '.pt')

if __name__ == "__main__":
    main()
