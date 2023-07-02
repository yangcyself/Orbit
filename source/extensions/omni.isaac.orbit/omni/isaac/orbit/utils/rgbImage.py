import numpy as np
import torch

class RGBImage:
    """An image helper class for storing and concatenating RGB images
    """
    def __init__(self, img):
        self.rgb_data = img

    def cat_horz(self, other):
        """Concatenate two images horizontally
        """
        #calculate the height difference
        height_diff = self.rgb_data.shape[0] - other.rgb_data.shape[0]
        #pad the smaller image with zeros
        if(height_diff > 0):
            other.rgb_data = np.pad(other.rgb_data, ((0, height_diff), (0,0), (0,0)), 'constant')
        elif(height_diff < 0):
            self.rgb_data = np.pad(self.rgb_data, ((0, -height_diff), (0,0), (0,0)), 'constant')
        self.rgb_data = np.concatenate((self.rgb_data, other.rgb_data), axis=1)
    
    def cat_vert(self, other):
        """Concatenate two images vertically
        """
        #calculate the width difference
        width_diff = self.rgb_data.shape[1] - other.rgb_data.shape[1]
        #pad the smaller image with zeros
        if(width_diff > 0):
            other.rgb_data = np.pad(other.rgb_data, ((0, 0), (0,width_diff), (0,0)), 'constant')
        elif(width_diff < 0):
            self.rgb_data = np.pad(self.rgb_data, ((0, 0), (0,-width_diff), (0,0)), 'constant')
        self.rgb_data = np.concatenate((self.rgb_data, other.rgb_data), axis=0)

    @property
    def shape(self):
        return self.rgb_data.shape

    @classmethod
    def from_torch(cls, tensor):
        """Create an image from a torch tensor
        """
        if(len(tensor.shape) == 4):
            tensor = tensor.squeeze(0)
        if(len(tensor.shape) == 2):
            tensor = tensor.unsqueeze(2)
        if(tensor.shape[2]==1):
            tensor = tensor.repeat(1,1,3)
        return cls(tensor.detach().to(torch.uint8).cpu().numpy())
    
    @staticmethod
    def horizontal_concat(img_list):
        """Concatenate a list of images horizontally
        """
        img = img_list[0]
        for i in range(1, len(img_list)):
            img.cat_horz(img_list[i])
        return img
    
    @staticmethod
    def vertical_concat(img_list):
        """Concatenate a list of images vertically
        """
        img = img_list[0]
        for i in range(1, len(img_list)):
            img.cat_vert(img_list[i])
        return img