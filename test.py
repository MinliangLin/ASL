import torch
from src.helper_functions.helper_functions import parse_args
from src.loss_functions.losses import AsymmetricLoss, AsymmetricLossOptimized
from src.models import create_model
import argparse
import tqdm

from PIL import Image
import numpy as np
from glob import glob
from time import time
from torch.utils.data import Dataset, DataLoader

parser = argparse.ArgumentParser(description='ASL Inference on a single image')

parser.add_argument('--model_path', type=str, default='./model/Open_ImagesV6_TRresNet_L_448.pth')
parser.add_argument('--glob', type=str, default='./pics/000000000885.jpg')
parser.add_argument('--model_name', type=str, default='tresnet_l')
parser.add_argument('--input_size', type=int, default=448)
parser.add_argument('--dataset_type', type=str, default='OpenImages')
parser.add_argument('--th', type=float, default=0.5)
parser.add_argument('--show', action="store_true", default=False)
parser.add_argument('--out', default=None)
parser.add_argument('--batch', type=int, default=300)
parser.add_argument('--cpu', action="store_true")
parser.add_argument('--num_workers', type=int, default=4)

class GlobDataset(Dataset):
    def __init__(self, glob_path, input_size=448, transform=None):
        self.input_size = input_size
        self.transform = transform
        self.globs = sorted(glob(glob_path))

    def __len__(self):
        return len(self.globs)

    def __getitem__(self, idx):
        img = Image.open(self.globs[idx]).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        img = img.resize((self.input_size, self.input_size))
        np_img = np.array(img, dtype=np.uint8)
        tensor_img = torch.from_numpy(np_img).permute(2, 0, 1).float() / 255.0  # HWC to CHW
        return (tensor_img, self.globs[idx])

def main():
    # parsing args
    args = parse_args(parser)
    device = torch.device("cuda" if not args.cpu else "cpu")
    loader = DataLoader(
        GlobDataset(args.glob, args.input_size),
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    # setup model
    state = torch.load(args.model_path, map_location='cpu')
    args.num_classes = state['num_classes']
    model = create_model(args).to(device)
    model.load_state_dict(state['model'], strict=True)
    model.eval()
    classes_list = np.array(list(state['idx_to_class'].values()))

    # doing inference
    def format_pred(np_output):
        detected_classes = classes_list[np_output > args.th]
        prob_output = np_output[np_output > args.th]
        str_output = ','.join(['{},{:.3f}'.format(c, p) for c, p in zip(detected_classes, prob_output)])
        return str_output

    start = time()
    final_output = []
    with torch.no_grad():
        for img_batch, path_batch in tqdm.tqdm(loader):
            pred_batch = torch.sigmoid(model(img_batch.to(device)))
            pred_batch = pred_batch.cpu().detach().numpy()
            output = '\n'.join([path+','+format_pred(pred) for pred, path in zip(pred_batch, path_batch)])
            if args.show:
                print(output)
            final_output.append(output)
    if args.out:
        with open(args.out, 'w') as f:
            f.write('\n'.join(final_output))

    print(f'time: {time()-start}')

if __name__ == '__main__':
    main()
