import os

def get_output_dir(args):
    output_dir = f"exps/{args.model.replace('/','-')}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir
