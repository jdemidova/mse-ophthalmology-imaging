from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode


def joint_resize(image, mask, size=(512, 512)):
    image = TF.resize(image, size, interpolation=InterpolationMode.BILINEAR)
    mask = TF.resize(mask, size, interpolation=InterpolationMode.NEAREST)
    return image, mask
