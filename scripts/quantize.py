import time
import argparse
import torch.quantization
import torch.nn.functional as F

from models import ResNet18

def run_torch_inference(model, x):
    # inference
    with torch.no_grad():
        # PyTorch inference
        torch_start = time.time()
        preds = model(x)
        print('Torch model predictions : ', preds)
        print('Torch model inference time : ', time.time()-torch_start)
        probs = F.softmax(preds, dim=1)

def main(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # create model and sample data
    model = ResNet18()
    model.load_state_dict(torch.load(args.input_pytorch_model, map_location=device))
    model.eval()
    x = torch.randn(1, 3, 32, 32, requires_grad=False)
    run_torch_inference(model, x)

    quantized_model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
    torch.save(quantized_model.state_dict(), args.output_quantized_model)

    quantized_model = ResNet18()
    quantized_model.load_state_dict(torch.load(args.input_pytorch_model, map_location=device))
    run_torch_inference(quantized_model, x)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Load documents from a directory into a Chroma collection")
    parser.add_argument("--input_pytorch_model", type=str,default="outputs/best.pth",help="Path to input pytorch model" )
    parser.add_argument("--output_quantized_model",type=str,default="outputs/best_quantized.pth",help="Path to output onnx model")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_arguments()
    main(args)