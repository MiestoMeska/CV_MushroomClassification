from torch import nn
import torchvision.transforms.functional as TF
class ResizeAndPad(nn.Module):
    def __init__(self, size, fill=0, padding_mode='constant'):
        super().__init__()
        self.size = size
        self.fill = fill
        self.padding_mode = padding_mode

    def forward(self, img):
        img = TF.resize(img, self.size)
        return img
