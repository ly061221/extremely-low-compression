import torch
from torchvision.models import vgg19
import lpips

vgg_model = vgg19(pretrained=True).features.eval()
for param in vgg_model.parameters():
    param.requires_grad = False

lpips_model = lpips.LPIPS(net='vgg')


# Load the LPIPS model
def lpips_distance(img1, img2, device):
    lpips_model.to(device)
    distance = lpips_model(img1, img2)
    return distance.item()


