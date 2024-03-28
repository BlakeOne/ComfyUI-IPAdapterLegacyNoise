import torch
import torchvision.transforms as TT

class IPAdapterLegacyNoise:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "strength": ("FLOAT", { "default": 0.25, "min": 0, "max": 1, "step": 0.05 }),                
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "make_noise"
    CATEGORY = "ipadapter"

    def make_noise(self, image, strength):
        image = image.permute([0,3,1,2])
        torch.manual_seed(0) # use a fixed random for reproducible results
        transforms = TT.Compose([
            TT.CenterCrop(min(image.shape[2], image.shape[3])),
            TT.Resize((224, 224), interpolation=TT.InterpolationMode.BICUBIC, antialias=True),
            TT.ElasticTransform(alpha=75.0, sigma=strength*3.5), # shuffle the image
            TT.RandomVerticalFlip(p=1.0), # flip the image to change the geometry even more
            TT.RandomHorizontalFlip(p=1.0),
        ])
        image = transforms(image.cpu())
        image = image.permute([0,2,3,1])
        image = image + ((0.25*(1-strength)+0.05) * torch.randn_like(image) )   # add further random noise

        return (image, )
