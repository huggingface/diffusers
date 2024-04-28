import torch 
import numpy as np    
     
class GaussingDistribution:      
    def __init__(self, parameters: torch.Tensor) -> None:    
        self.mean, log_variance = torch.chunk(parameters, 2, dim=1) 
        self.log_variance = torch.clamp(log_variance, -30.0, 20.0)   
        self.std = torch.exp(0.5 * self.log_variance)
    
    def sample(self): 
        return self.mean + self.std * torch.rand_like(self.std) 

class DenoisingDiffusionProbabilisticModelSampler:
    def __init__(self, 
                 generator: torch.Generator, 
                 number_training_steps = 1000, 
                 beta_start: float = 0.00085, 
                 beta_end: float = 0.0120) -> None:
                     
        self.betas = torch.linspace(beta_start ** 0.5, beta_end ** 0.5, number_training_steps, dtype=torch.float32) ** 2 
        self.alphas = 1.0 - self.betas
        self.alphas_cumlative_product = torch.cumprod(self.alphas, d_model = 0)
        self.one = torch.tensor(1.0)
        self.generator = generator
        self.number_training_timesteps = number_training_steps
        self.timesteps = torch.from_numpy(np.arange(0, number_training_steps)[::-1].copy())
    
    def set_inference_timesteps(self, number_infernece_steps = 50):
        self.number_inference_steps = number_infernece_steps
        ratio = self.number_training_timesteps // self.number_inference_steps
        timesteps = (np.arange(0, number_infernece_steps) * ratio).round()[::-1].copy().astype(np.int64)
        self.timesteps = torch.from_numpy(timesteps)
    
    def _get_previous_timestep(self, timestep: int) -> int:
        previous_step = timestep - self.number_training_timesteps // self.number_inference_steps
        return previous_step
    
    def _get_variance(self, timestep: int) -> torch.Tensor:
        previous_step = self._get_previous_timestep(timestep)
        alphas_product_timestep = self.alphas_cumlative_product[timestep]
        alphas_product_timestep_previous = self.alphas_cumlative_product[previous_step] if previous_step >=0 else self.one 
        current_beta_timestep = 1 - alphas_product_timestep / alphas_product_timestep_previous
        variance = (1 - alphas_product_timestep_previous) / (1 - alphas_product_timestep) * current_beta_timestep
        variance = torch.clamp(variance, 1e-20)
        return variance 
    
    def set_strength(self, strength = 1):
        """
        Set how much noise to add to the input image. 
        More noise (strength ~ 1) means that the output will be further from the input image.
        Less noise (strength ~ 0) means that the output will be closer to the input image.
        """
        startstep = self.number_inference_steps - int(self.number_inference_steps * strength)
        self.timesteps = self.timesteps[startstep:]
        self.startstep = startstep
    
    def step(self, timestep: int, 
             latents: torch.Tensor, 
             model_output: torch.Tensor):
                 
        t = timestep
        previous_step = self._get_previous_timestep(t)
        alphas_product_timestep = self.alphas_cumlative_product[t]
        alphas_product_timestep_previous = self.alphas_cumlative_product[previous_step] if previous_step >=0 else self.one 
        current_alphas_product_timestep = alphas_product_timestep / alphas_product_timestep_previous
        beta_timestep = 1 - alphas_product_timestep
        beta_timestep_previous = 1 - alphas_product_timestep_previous
        current_beta_timestep = 1 - current_alphas_product_timestep
        predict_original_samples = (latents - beta_timestep ** (0.5) * model_output) / alphas_product_timestep
        predict_original_samples_coeff = (alphas_product_timestep_previous ** (0.5) * current_beta_timestep) / beta_timestep
        predict_current_samples_coeff = current_alphas_product_timestep ** (0.5) * beta_timestep_previous / beta_timestep
        predict_previous_samples = predict_original_samples_coeff * predict_original_samples + predict_current_samples_coeff * latents
        variance = 0
        if t > 0:
            device = model_output.device 
            dtype = model_output.dtype 
            noise = torch.randn(model_output.shape, generator=self.generator, device=device, dtype=dtype)
            variance = (self._get_variance(t) ** 0.5) * noise 
        predict_previous_samples = predict_previous_samples + variance
        return predict_previous_samples
    
    def add_noise(self, 
                  original_samples: torch.FloatTensor, 
                  timestep: torch.IntTensor) -> torch.FloatTensor:
                      
        alphas_cumlative_product = self.alphas_cumlative_product.to(device = original_samples.device, dtype = original_samples.dtype)
        timestep = timestep.to(original_samples.device)
        alphas_product_timestep_squaroot = alphas_cumlative_product[timestep] ** 0.5
        alphas_product_timestep_squaroot = alphas_product_timestep_squaroot.flatten()
        while len(alphas_product_timestep_squaroot.shape) < len(original_samples.shape):
            alphas_product_timestep_squaroot = alphas_product_timestep_squaroot.unsqueeze(-1)

        alphas_product_timestep_squaroot_mins = (1- alphas_cumlative_product[timestep]) ** 0.5 
        alphas_product_timestep_squaroot_mins = alphas_product_timestep_squaroot_mins.flatten()
        while len(alphas_product_timestep_squaroot_mins.shape) < len(original_samples.shape):
            alphas_product_timestep_squaroot_mins = alphas_product_timestep_squaroot_mins.unsqueeze(-1)

        device = original_samples.device
        dtype = original_samples.dtype 
        noise = torch.randn(original_samples.shape, generator=self.generator, device=device, dtype=dtype)
        noisy_samples = alphas_product_timestep_squaroot * original_samples + alphas_product_timestep_squaroot_mins * noise
        return noisy_samples 
