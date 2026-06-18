# Copyright (C) 2026 Boogu Team.

import torch


class SimpleRMSNorm(torch.nn.Module):
    """
    Simple RMS Normalization implementation using native PyTorch operations.

    This is a pure PyTorch implementation that matches the functionality of RMSNorm
    but without Triton optimizations. Useful for debugging, testing, or when Triton
    is not available.

    Args:
        hidden_size: The size of the hidden dimension
        eps: A small value added to the denominator for numerical stability
        dropout_p: Dropout probability (applied before normalization)
        zero_centered_weight: If True, initialize weight to zeros instead of ones
        device: Device to place the parameters on
        dtype: Data type for the parameters
    """

    def __init__(
        self,
        hidden_size,
        eps=1e-5,
        dropout_p=0.0,
        zero_centered_weight=False,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.eps = eps
        self.hidden_size = hidden_size

        # Dropout layer (same as RMSNorm)
        if dropout_p > 0.0:
            self.drop = torch.nn.Dropout(dropout_p)
        else:
            self.drop = None

        self.zero_centered_weight = zero_centered_weight

        # Weight parameter (same as RMSNorm)
        self.weight = torch.nn.Parameter(torch.zeros(hidden_size, **factory_kwargs))

        # No bias in RMS normalization (same as RMSNorm)
        self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize parameters (same logic as RMSNorm)"""
        if not self.zero_centered_weight:
            torch.nn.init.ones_(self.weight)
        else:
            torch.nn.init.zeros_(self.weight)

    def _simple_rms_norm(self, x, weight, eps=1e-5, zero_centered_weight=False):
        """
        Simple RMS normalization implementation using native PyTorch.

        Args:
            x: Input tensor [..., hidden_size]
            weight: Weight parameter [hidden_size]
            eps: Small value for numerical stability
            zero_centered_weight: If True, add 1.0 to weight

        Returns:
            Normalized tensor with same shape as input
        """
        # Convert to float32 for numerical stability (like the reference implementation)
        input_dtype = x.dtype
        x = x.float()
        weight = weight.float()

        # Apply zero-centered weight transformation if needed
        if zero_centered_weight:
            weight = weight + 1.0

        # Compute RMS normalization

        # Compute mean of squared values along the last dimension
        variance = x.pow(2).mean(dim=-1, keepdim=True)

        # Compute reciprocal standard deviation (rstd)
        rstd = torch.rsqrt(variance + eps)  # 1 / sqrt(variance + eps)

        # Apply normalization and scaling
        normalized = x * rstd * weight

        # Convert back to original dtype
        return normalized.to(input_dtype)

    def forward(self, x, residual=None, prenorm=False, residual_in_fp32=False):
        """
        Forward pass matching the interface of RMSNorm.

        Args:
            x: Input tensor
            residual: Optional residual tensor to add before normalization
            prenorm: If True, return both normalized output and residual
            residual_in_fp32: If True, compute residual in fp32

        Returns:
            If prenorm=False: normalized tensor
            If prenorm=True: (normalized tensor, residual tensor)
        """
        # Store original shape and dtype
        orig_shape = x.shape
        orig_dtype = x.dtype

        # Handle empty tensors (edge case)
        if x.numel() == 0:
            if prenorm:
                residual_out = torch.empty_like(x, dtype=torch.float32 if residual_in_fp32 else x.dtype)
                return x, residual_out
            return x

        # Reshape to 2D for processing (batch_size * seq_len, hidden_size)
        x_2d = x.view(-1, x.shape[-1])

        # Apply dropout if enabled and in training mode
        if self.drop is not None and self.training:
            x_2d = self.drop(x_2d)

        # Add residual if provided
        if residual is not None:
            # Ensure residual has the same shape as input
            if residual.shape != orig_shape:
                raise ValueError(f"Residual shape {residual.shape} doesn't match input shape {orig_shape}")

            residual_2d = residual.view(-1, residual.shape[-1])

            # Convert to appropriate dtype for residual computation
            if residual_in_fp32:
                x_2d = x_2d.float()
                residual_2d = residual_2d.float()

            # Add residual
            x_2d = x_2d + residual_2d

        # Store residual for prenorm case
        if prenorm:
            if residual_in_fp32:
                residual_out = x_2d.float()
            else:
                residual_out = x_2d.to(orig_dtype)

        # Apply RMS normalization
        normalized_2d = self._simple_rms_norm(x_2d, self.weight, self.eps, self.zero_centered_weight)

        # Reshape back to original shape
        normalized = normalized_2d.view(orig_shape)

        # Return based on prenorm flag
        if prenorm:
            residual_out = residual_out.view(orig_shape)
            return normalized, residual_out
        else:
            return normalized
