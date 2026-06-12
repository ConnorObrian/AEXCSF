from enum import Enum
import os
from torch.utils.data import Dataset
import warnings
import torch
from PIL import Image
import torchvision
import torchvision.transforms.v2 as v2
from torchvision.transforms.functional import to_tensor

from .fdupes import find_duplicate_files, remove_duplicate_files

class ScalerKind(str, Enum):
    MINMAX = "minmax"
    ROLLING_MINMAX = "rolling_minmax"
    CONSTANT = "constant"


class MinMaxNormalize:
    def __init__(self, per_channel=True, epsilon=1e-8):
        """
        Args:
            per_channel (bool): If True, normalize each channel independently.
                                If False, normalize across all channels together.
        """
        self.per_channel = per_channel
        self.epsilon = epsilon

    def __call__(self, tensor):
        if self.per_channel:
            # Normalize each channel independently
            for c in range(tensor.shape[0]):  # Loop over channels
                x_min = tensor[c].min()
                x_max = tensor[c].max()
                tensor[c] = (tensor[c] - x_min) / (x_max - x_min + self.epsilon)
        else:
            # Normalize across all channels together
            x_min = tensor.min()
            x_max = tensor.max()
            tensor = (tensor - x_min) / (x_max - x_min + self.epsilon)
        
        return tensor

# TODO: Test whether this does not harm performance (but it shouldn't)
class ConstantMinMaxScaler:
    def __init__(self,  min:float, max:float, epsilon=1e-8,):
        """
        Scales the values of a tensor_like/array_like in the range [0, 1].
        It assumes that the max and min values are known. 
        Which can be assumed for images for example.
        
        :param self: Description
        :param min: Description
        :type min: float
        :param max: Description
        :type max: float
        :param epsilon: Description
        """
        self.epsilon = epsilon
        self.min = min
        self.max = max

    def __call__(self, tensor_like):
        res = (tensor_like - self.min) / (self.max - self.min + self.epsilon)
        return res

class MinMaxRollingObsScaler:
    def __init__(self, epsilon=1e-8, min:float=torch.inf, max:float=-torch.inf):
        """
        Rolling rescaler that tracks minimum and maximum which makes sense for the using samples in the latent representation 
        of Conv AEs, VAEs and VQ-VAEs as minima and maxima are not known a-priori and might be different per environment.

        Args:
            epsilon (float): Minimal epsilon to prevent division by zero.
            min (float): So far known minimum of the tracked tensors.
            max (float): So far known maximum of the tracked tensors.
        """
        self.epsilon = epsilon
        self.min = min
        self.max = max
    
    def __call__(self, tensor_like):
        if self.min > (min_cand:=tensor_like.min()):
            self.min = min_cand

        if self.max < (max_cand:=tensor_like.max()):
            self.max = max_cand
        res = (tensor_like - self.min) / (self.max - self.min + self.epsilon)
        return res

class RLDataset(Dataset):
    """class to parse the dataset and return the data in a format that can be used by the model
    Args:
        Dataset (torch.utils.data.Dataset): torch Dataset class
        path (str): path to the dataset
        eliminate_duplicates (bool, optional): whether to eliminate duplicate data. Defaults to False.
        transform ([type], optional): transformation used on the images. Defaults to None.
        image_file_type (str, optional): defines the image file type to be found inside the path. Defaults to "jpg".
        image_size (list[int, int, int], optional): defines the output size of the image. Defaults to [3, 224, 224].
    """
    def __init__(self, root_dir : str, eliminate_duplicates : bool = False, transform = None, image_file_type : str = "jpg", image_size : list[int, int, int] = [3, 224, 224], clamp : bool = True, legacy_input_scaling: bool = False) -> None:
        self.root_dir = os.path.abspath(root_dir)
        self.eliminate_duplicates = eliminate_duplicates
        self.transform = transform
        self.image_file_type = image_file_type
        self.dir = [f for f in os.listdir(self.root_dir) if f.endswith(self.image_file_type)]
        if self.eliminate_duplicates:
            self._eliminate_duplicates()
        preprocess_steps = [v2.Resize(image_size[1:])]
        if image_size[0] == 1:
            preprocess_steps.append(v2.Grayscale(num_output_channels=1))
        if clamp and legacy_input_scaling:
            preprocess_steps.append(MinMaxNormalize(per_channel=False))
            preprocess_steps.append(v2.ToDtype(torch.float32, scale=True))
        self.preprocess = v2.Compose(preprocess_steps)
        self._check_dataset_for_nans()
        
    def __len__(self):
        return len(self.dir)
    
    def _check_dataset_for_nans(self):
        for imgpath in self.dir:
            img = to_tensor(Image.open(os.path.join(self.root_dir, imgpath)))
            if torch.isnan(img).any():
                print(f"Image {img} contains NaN values, deleting this image.")
                os.remove(os.path.join(self.root_dir, imgpath))
        self.dir = [f for f in os.listdir(self.root_dir) if f.endswith(self.image_file_type)]
                
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        img_name = self.dir[idx]
        im_path = os.path.join(self.root_dir, img_name)
        image = to_tensor(Image.open(im_path))
        
        image = self.preprocess(image)
        if torch.isnan(image).any():
            warnings.warn(f"Image {img_name} contains NaN values, consider deleting the file. Replacing it with 0 for now.")
            image[image != image] = 0
               
        if self.transform:
            image = self.transform(image)
        
        return image
    
    def _eliminate_duplicates(self):
        """eliminate duplicate images from the dataset using hashes
        """
        warnings.warn("This method is experimental and may not work as expected, also it needs a lot of computational power as well as disk usage, so use with caution.")
        duplicates = find_duplicate_files(self.root_dir)
        remove_duplicate_files(duplicates)
        self.dir = [f for f in os.listdir(self.root_dir) if f.endswith(self.image_file_type)]
        

def train_val_dataset(dataset, val_split=0.25):
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [1-val_split, val_split])
    return train_dataset, test_dataset
