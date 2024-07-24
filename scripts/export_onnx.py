import onnx
import time
import torch
import argparse
import onnxruntime as ort
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
    
def export_onnx(model, x):
    torch.onnx.export(model,                      # model being run
                      x,                          # model input (or a tuple for multiple inputs)
                      args.output_onnx_model,     # where to save the model (can be a file or file-like object)
                      export_params=True,         # store the trained parameter weights inside the model file
                      opset_version=10,           # the ONNX version to export the model to
                      do_constant_folding=True,   # whether to execute constant folding for optimization
                      input_names = ['input'],    # the model's input names
                      output_names = ['output'],  # the model's output names
                      dynamic_axes={'input' : {0 : 'batch_size'},       # variable length axes
                                    'output' : {0 : 'batch_size'}})

    # check converted onnx model
    onnx_model = onnx.load(args.output_onnx_model)
    onnx.checker.check_model(onnx_model)

def main():

    # create model and sample data
    model = ResNet18()
    model.load_state_dict(torch.load(args.input_pytorch_model))
    model.eval()
    x = torch.randn(1, 3, 32, 32, requires_grad=False)

    run_torch_inference(model, x)

    # Export the Pytorch model to ONNX
    export_onnx(model, x)

    # Create an ONNX Runtime session and run inference
    ort_session = ort.InferenceSession(args.output_onnx_model)
    onnx_start = time.time()
    outputs = ort_session.run(None, {'input': x.numpy()})
    print('Onnx model output:', outputs)
    print('Onnx model inference time :',  time.time()-onnx_start)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Load documents from a directory into a Chroma collection")
    parser.add_argument("--input_pytorch_model", type=str,default="outputs/best.pth",help="Path to input pytorch model" )
    parser.add_argument("--output_onnx_model",type=str,default="outputs/best.onnx",help="Path to output onnx model",)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_arguments()
    main()