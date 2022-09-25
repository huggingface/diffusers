from lpips_utils import LPIPSLoss


class LPIPSCallback:
    def __init__(self, loss_fnc, target_img, device="cuda"):
        self.loss = loss_fnc.to(device)
        self.device = device
        self.target_img = target_img.squeeze().unsqueeze(0).float().to(device)
        
    def __call__(self, lpips_img):
        start_device = lpips_img.device
        lpips_img = lpips_img.squeeze().unsqueeze(0).float().to(self.device)
        return self.loss(lpips_img, self.target_img).to(start_device)
        

class NormalDistLoss:
    def __call__(self, latent):
        # mean should be 0
        loss = latent.mean() ** 2
        # std should be 1
        loss += (latent.std() - 1) ** 2
        return loss
    

class ContrastLoss:
    def __call__(self, images):
        if images.ndim == 3:
            images = images.unsqueeze(0)
        # flatten into [batch size, channels, N]
        flat = images.reshape(images.shape[0], images.shape[1], -1)
        loss = flat.std(dim=-1).mean()
        return loss
    
# TODO: Write colour consistency loss!

lpips_loss = LPIPSLoss(lpips_net="squeeze", # squeeze 
                               batch_size=2,
                               use_aug_assuming_equal_img_shape=False)

def create_callbacks(use_lpips=False, lpips_image=None, use_normal_dist=False, use_contrast=False,
                     lpips_frequency=5, contrast_frequency=5, lpips_lr=100, contrast_lr=100,
                     normal_dist_lr=0.1): 
    callbacks = []
    if use_lpips and lpips_image is not None:
        lpips_callback = {"loss_function": LPIPSCallback(lpips_loss, lpips_image, device="cuda"),
                           "weight": 1,
                           "frequency": lpips_frequency,
                           "lr": lpips_lr,
                           "apply_to_image": True}
        callbacks.append(lpips_callback)
    if use_normal_dist:
        lpips_callback = {"loss_function": NormalDistLoss(),
                           "weight": 1,
                           "frequency": 2,
                           "lr": normal_dist_lr,
                           "apply_to_image": False}
        callbacks.append(lpips_callback)
    if use_contrast:
        lpips_callback = {"loss_function": ContrastLoss(),
                           "weight": 1,
                           "frequency": contrast_frequency,
                           "lr": contrast_lr,
                           "apply_to_image": True}
        callbacks.append(lpips_callback)
    return callbacks