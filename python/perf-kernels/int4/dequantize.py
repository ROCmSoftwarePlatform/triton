import torch
import triton
import triton.language as tl

@triton.jit
def dequantize_int4(x_ptr,
                    scale_ptr,
                    shift_ptr,
                    n_rows,
                    n_cols: tl.constexpr, 
                    x_out_dequant_ptr,
                    BLOCK_SZ: tl.constexpr,
                    PACKED_PER_VAL: tl.constexpr,
                    ):

    #Get PID
    pid = tl.program_id(0)

    #Load x_ptr values
    row_start = pid*BLOCK_SZ
    offset = (row_start + tl.arange(0, BLOCK_SZ))[:, None]*n_cols + tl.arange(0, n_cols)[None, :]
    mask = ((row_start + tl.arange(0, BLOCK_SZ))[:, None] < n_rows) & (tl.arange(0, n_cols)[None, :] < n_cols)
    x = tl.load(x_ptr + offset, mask=mask)
    #print(f"pid={pid}: x={x} ")

    #Shift each packed value in the input into separate column
    shifts = tl.arange(0, PACKED_PER_VAL) * 4
    shifts = tl.flip(shifts, dim=0) #Arange doesn't go in reverse order, so flip here
    quant_offset = (
        x[:, :, None] >> shifts 
    )  # (num_rows, num_cols, PACKED_PER_VAL)
    quant_offset = quant_offset & 0xF #This is needed for removing the top half bits in column with offset 0.  
    quant_offset = tl.reshape(
        quant_offset, (BLOCK_SZ, n_cols* PACKED_PER_VAL)
    )
    #print(f"pid={pid}: quant_offset={quant_offset} ")
    
    #Convert to fp16.
    #Note: Instead of converting int4 to float16 view it as float16 and 
    #then multiply by 2*14(16384.0) * 2^10(1024). 1 in int4(b0001) will be a subnormal number 
    #in fp16(b0-00000-0000000001)=2^-14*2^-10, so we multiply by 
    #Source: https://github.com/facebookresearch/xformers/blob/36464229859a177a165d142db788db45cbe6b272/xformers/ops/fmha/triton_splitk.py#L704
    quant_offset = (quant_offset & 0xF).to(tl.uint16).to(tl.float16, bitcast=True)
    quant_offset = (quant_offset * 16384.0).to(tl.float16)
    #print(f"pid={pid}: quant_offset={quant_offset} ")

    scale = tl.load(scale_ptr + row_start + tl.arange(0, BLOCK_SZ))
    shift = tl.load(shift_ptr + row_start + tl.arange(0, BLOCK_SZ))
    #print(f"pid={pid}: scale={scale}, shift={shift} ")

    #Dequantize
    scale_1024 = scale * 1024
    dequant = quant_offset * scale_1024 + shift

    #Write out dequantized values. Note: we have twice the number of values now.
    offset = (row_start + tl.arange(0, BLOCK_SZ))[:, None]*n_cols*2 + tl.arange(0, n_cols*2)[None, :]
    mask = ((row_start + tl.arange(0, BLOCK_SZ))[:, None] < n_rows) & (tl.arange(0, n_cols*2)[None, :] < n_cols*2)
    tl.store(x_out_dequant_ptr + offset, dequant, mask=mask)


if __name__ == '__main__':
    '''
    data = [[240, 130, 33, 1]]
    scale = [[7.097]]
    shift = [[4]]

    data = [[1, 113, 244,  20]]
    scale = [[6.008]]
    shift = [[0]]
    '''

    data = [[240, 130, 33, 1],
            [1, 113, 244,  20]]
    scale = [[7.097],
            [6.008]]
    shift = [[4],
            [0]]
    x = torch.tensor(data, dtype=torch.uint8).to(device="cuda")
    print(f"x={x}")
    #print(x.storage().nbytes())
    scale = torch.tensor([scale],dtype=torch.float16).to(device="cuda")
    shift = torch.tensor([shift],dtype=torch.uint8).to(device="cuda")
    print(f"scale={scale}")
    print(f"shift={shift}")

    x_dequant = torch.zeros((x.shape[0],x.shape[-1]*2), dtype=torch.float16).to(device="cuda")

    grid = lambda meta: (triton.cdiv(x.shape[0], meta['BLOCK_SZ']),)
    dequantize_int4[grid](x, scale, shift, x.shape[0], x.shape[1], x_dequant, BLOCK_SZ=1, PACKED_PER_VAL=2)
    print(f"x_dequant={x_dequant}")