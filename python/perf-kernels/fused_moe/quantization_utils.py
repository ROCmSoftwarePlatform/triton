import torch

INT8_MAX = 127

fp8_max_repr_val_dict = {torch.float8_e4m3fnuz: 240.0, torch.float8_e4m3fn: 448.0, torch.float8_e5m2fnuz: 57344.0, torch.float8_e5m2: 57344.0}

def quantize_int8(tensor: torch.Tensor, dim=None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    max_vals = tensor.abs().amax(dim=[i for i in range(tensor.dim()) if i != dim], keepdim=True)

    # Avoid division by zero
    max_vals[max_vals == 0] = 1e-8

    # Compute scale factors for each channel
    scale = INT8_MAX / max_vals.to(torch.float32)

    # Quantize the tensor
    tensor = tensor * scale
    tensor = tensor.round_()
    tensor.clamp_(-INT8_MAX, INT8_MAX)
    tensor_quantized = tensor.to(torch.int8)

    return tensor_quantized, scale, 1 / scale

# TODO Investigate if fnuz works
def quantize_fp8(tensor: torch.Tensor, dim: tuple=None, fp8_type = torch.float8_e4m3fnuz) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    fp8_max_repr_val = fp8_max_repr_val_dict[fp8_type]
    max_vals = tensor.abs().amax(dim=[i for i in range(tensor.dim()) if i != dim], keepdim=True)

    # Avoid division by zero
    max_vals[max_vals == 0] = 1e-8

    # Compute scale factors for each channel
    scale = fp8_max_repr_val / max_vals.to(torch.float32)

    # Quantize the tensor
    tensor = tensor * scale
    tensor = tensor.round_()
    tensor.clamp_(-fp8_max_repr_val, fp8_max_repr_val)
    tensor_quantized = tensor.to(fp8_type)

    return tensor_quantized, scale, 1 / scale
