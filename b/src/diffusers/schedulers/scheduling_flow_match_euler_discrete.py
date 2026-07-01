        self._shift = shift

        self.sigmas = sigmas.to("cpu")  # to avoid too much CPU/GPU communication
        self.sigma_min = unshifted_sigmas[-1].item()
        self.sigma_max = unshifted_sigmas[0].item()

    @property
    def shift(self):
        timesteps = torch.from_numpy(timesteps).to(dtype=torch.float32)

        sigmas = timesteps / num_train_timesteps

        # Compute sigma_min/sigma_max from the unshifted sigmas so that set_timesteps()
        # does not apply timestep shifting twice -- it already applies shift/recomputation
        # from sigma_min/sigma_max.
        unshifted_sigmas = sigmas.clone()

        if not use_dynamic_shifting:
            # when use_dynamic_shifting is True, we apply the timestep shifting on the fly based on the image resolution
            sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)
