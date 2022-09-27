import torch
import lpips
import torch.nn.functional as F


class MakeCutoutsRHW(torch.nn.Module):
    def __init__(self, cut_size, cut_pow=1.):
        super().__init__()
        self.cut_size = cut_size
        self.cut_pow = cut_pow

    def forward(self, x, params=None):
        sideY, sideX = x.shape[2:4]
        max_size = min(sideX, sideY)
        min_size = min(sideX, sideY, self.cut_size)
        if params is None:
            size_frac = torch.rand(1) ** self.cut_pow
            offsetx_frac = torch.rand(1)
            offsety_frac = torch.rand(1)
            self.params = size_frac, offsetx_frac, offsety_frac
        else:
            size_frac, offsetx_frac, offsety_frac = params
        size = int(size_frac * (max_size - min_size) + min_size)
        offsetx = ((sideX - size + 1) * offsetx_frac).round().int()
        offsety = ((sideY - size + 1) * offsety_frac).round().int()
        
        #offsetx = torch.randint(0, sideX - size + 1, ())
        #offsety = torch.randint(0, sideY - size + 1, ())
        cutout = x[:, :, offsety:offsety + size, offsetx:offsetx + size]
        resized_cutout = F.adaptive_avg_pool2d(cutout, self.cut_size)
        return resized_cutout
    
    
class LPIPSLoss:
    def __init__(self, lpips_net="vgg", batch_size=1, use_aug_assuming_equal_img_shape=False):
        self.use_aug_assuming_equal_img_shape = use_aug_assuming_equal_img_shape
        self.lpips_transform = MakeCutoutsRHW(224)
        self.lpips_loss_fn = lpips.LPIPS(net=lpips_net)
        self.batch_size = batch_size
        
    def __call__(self, img, lpips_img, device="cuda"):
        if self.use_aug_assuming_equal_img_shape:
            stacked = torch.cat([img, lpips_img], dim=1)
            augmented = torch.cat([self.lpips_transform(stacked) for _ in range(self.batch_size)])
            # shape: [bs, 6, size, size] size=224
            aug_img = augmented[:, 0:3]
            aug_target_img = augmented[:, 3:] 
        else:
            self.lpips_transform.params = None
            aug_img = []
            aug_target_img = []
            for _ in range(self.batch_size):
                aug_out = self.lpips_transform(img)
                aug_target = self.lpips_transform(lpips_img, params=self.lpips_transform.params)
                aug_img.append(aug_out)
                aug_target_img.append(aug_target)
            aug_img = torch.cat(aug_img, dim=0)
            aug_target_img = torch.cat(aug_target_img, dim=0)
                
        lpips_loss = self.lpips_loss_fn.forward(aug_img, aug_target_img.to(device), normalize=True)
        lpips_loss = lpips_loss.mean().squeeze()
        return lpips_loss
    
    def to(self, *args, **kwargs):
        self.lpips_loss_fn.to(*args, **kwargs)
        return self