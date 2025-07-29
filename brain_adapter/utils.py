import os
import random
import argparse
import pathlib
import gc
import numpy as np
import wandb
from scipy.stats import pearsonr
import torch
from torchvision import transforms

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def clear_cuda_memory():
    """
    Clears GPU memory by deleting variables, running garbage collection,
    and calling torch.cuda memory cleanup functions.
    """
    # Python garbage collection
    gc.collect()

    if torch.cuda.is_available():
        # Clear PyTorch's internal cache
        torch.cuda.empty_cache()
        # Release cached memory held by CUDA driver
        torch.cuda.ipc_collect()

class str2bool(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        if values.lower() in ('true', 't', '1'):
            setattr(namespace, self.dest, True)
        elif values.lower() in ('false', 'f', '0'):
            setattr(namespace, self.dest, False)
        else:
            raise argparse.ArgumentTypeError(f"Invalid value for {self.dest}: {values}")
        
def get_preprocess(shape=(434, 434), mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    return transforms.Compose([
        transforms.ToPILImage(),               # Convert tensor to PIL image
        transforms.Resize(shape),       # Resize image
        transforms.ToTensor(),                 # Convert PIL to tensor in [0, 1]
        transforms.Normalize(mean=mean, std=std)  # Normalize using ImageNet stats or custom
    ])

def compute_pearson_correlation(pred, target):
    """
    Compute Pearson correlation between predicted and target tensors using SciPy.
    
    Args:
        pred: Predicted tensor of shape (N, fmri_features) or (N, ..., fmri_features)
        target: Target tensor of shape (N, fmri_features) or (N, ..., fmri_features)
        
    Returns:
        mean_correlation: Mean correlation across all images
        per_image_correlation: Array of correlations for each image (shape: [N])
    """
    # If input is a torch tensor, convert to NumPy.
    if torch.is_tensor(pred):
        pred = pred.detach().cpu().numpy()
    if torch.is_tensor(target):
        target = target.detach().cpu().numpy()
    
    # Here, we assume the inputs are of shape (N, fmri_features)
    N = pred.shape[0]
    correlations = []
    for i in range(N):
        r, _ = pearsonr(pred[i], target[i])
        correlations.append(r)
    correlations = np.array(correlations)
    mean_correlation = correlations.mean()
    return mean_correlation, correlations
                
def save_checkpoint_accelerator(state, accelerator, args):
    '''
    /brain_mapping/model_weights/
    |__dino-vitb14
    |    |__xxx.pth.tar
    |...
    ''' 
    
    save_dir = os.path.join(args.model_weights_dir, args.args.model_name)
    pathlib.Path(save_dir).mkdir(parents=True, exist_ok=True) # "/brain_mapping/model_weights/dino-vitb14/"
    
    filename = os.path.join(save_dir, state['epoch'] + ".pth.tar")
    accelerator.save(state, filename)

def logging_accelerator(loggers, values, batch_size, accelerator, args, var_names=None, status="Train", isLastBatch=False):
    pairs = {}
    if not var_names: var_names = [None] * len(values)
    for logger, value, var_name in zip(loggers, values, var_names):
        val = value.detach().item() if isinstance(value, torch.Tensor) else value
        logger.update(val, batch_size)
        if var_name: pairs[var_name] = val

    if status == "Training" or (status == "Validation" and isLastBatch):
        if accelerator.is_main_process and args.wandb and var_names: # just update values on the main process
            for var_name, value in zip(var_names, values):
                wandb.log(pairs)

class ProgressMeterAcc(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch, accelerator):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        if accelerator:
            accelerator.print('  '.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

    def synchronize_between_processes(self, accelerator):
        for meter in self.meters:
            meter.synchronize_between_processes(accelerator)
    
class AverageMeterAcc(object):
    """Computes and stores the average and current value (Accelerator compatible)"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0.0

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

    def synchronize_between_processes(self, accelerator):
        c = accelerator.reduce(torch.tensor(self.count, dtype=torch.float32, device=accelerator.device), reduction="sum")
        s = accelerator.reduce(torch.tensor(self.sum, dtype=torch.float32, device=accelerator.device), reduction="sum")
        self.count = int(c.item())
        self.sum = s.item()
        self.avg = self.sum / self.count if self.count != 0 else 0.0
