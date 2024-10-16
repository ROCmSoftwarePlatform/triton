import torch
import triton
import triton.language as tl


@triton.jit
def quantize_int4(x_ptr,
                scale_out_ptr,
                shift_out_ptr,
                x_out_quant_ptr,
                n_rows,
                n_cols: tl.constexpr, 
                BLOCK_SZ: tl.constexpr,
                ):
    
    #Pid
    pid = tl.program_id(0)
    #print(f"pid={pid}")

    #Load x_ptr values
    row_start = pid*BLOCK_SZ
    offset = (row_start + tl.arange(0, BLOCK_SZ))[:, None]*n_cols + tl.arange(0, n_cols)[None, :]
    mask = ((row_start + tl.arange(0, BLOCK_SZ))[:, None] < n_rows) & (tl.arange(0, n_cols)[None, :] < n_cols)
    #print(f"pid={pid}: offset={offset} ")
    x = tl.load(x_ptr + offset, mask=mask)

    #Find max and min
    xmax = tl.max(x,axis=1)
    xmin = tl.min(x,axis=1)
    #print(f"pid={pid}: max={max} min={min}")
    
    #Calculate scale
    scale = (xmax - xmin) / 15 #Total number of values that can be represented by INT4

    #Calculate shift/zero point. Note: shift is negative of zero-point and not yet scaled
    shift = xmin
    shift = shift.to(tl.uint8)
    #print(f"pid={pid}: scale={scale} shift={shift}")

    #quantize 
    x_quant = (x - shift.reshape(BLOCK_SZ,1))/scale.reshape(BLOCK_SZ,1) + 0.5 
    x_quant = x_quant.to(tl.uint8) 
    x_quant = min(x_quant, 15)

    #x_quant = x_quant & 0xF #before this
    #print(f"pid={pid}: x_quant={x_quant} ")

    #write out shift and scale
    tl.store(scale_out_ptr + pid + tl.arange(0,BLOCK_SZ), scale)
    tl.store(shift_out_ptr + pid + tl.arange(0,BLOCK_SZ), shift)

    #write out quantized values
    tl.store(x_out_quant_ptr + offset, x_quant, mask=mask)


if __name__ == '__main__':
    #data = [[0, 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],[12,15,22,34,36,57,60,70,85,91,92,120,130,135,145,150]]
    #data = [[12,15,22,34,36,57,60,70,85,91,92,120,130,135,145,150]]
    data = [[110.51, 4.05, 60.45, 18.76, 20.10, 13.28, 4.20, 11.20]]
    #data = [[110.51, 4.05, 60.45, 18.76, 20.10, 13.28, 4.20, 11.20],
    #        [0.0, 4.05, 40.35, 8.26, 90.10, 23.28, 4.20, 26.20]]
    x = torch.tensor(data, dtype=torch.float16).to(device="cuda")
    print(f"input x={x}")
    #print(x.storage().nbytes())
    scale = torch.zeros((x.shape[0]),dtype=torch.float16).to(device="cuda")
    shift = torch.zeros((x.shape[0]),dtype=torch.uint8).to(device="cuda")

    x_quant = torch.zeros((x.shape[0],x.shape[-1]), dtype=torch.uint8).to(device="cuda")

    grid = lambda meta: (triton.cdiv(x.shape[0], meta['BLOCK_SZ']),)
    quantize_int4[grid](x, scale, shift, x_quant, x.shape[0], x.shape[1], BLOCK_SZ=2)
    print(f"x_quant={x_quant}")
    #print(x_quant.storage().nbytes())
    print(f"scale={scale}")
    #print(scale.storage().nbytes())
    print(f"shift={shift}")
    #print(shift.storage().nbytes())
    x_quant_packed = x_quant.reshape(-1,)[::2] << 4 | x_quant.reshape(-1,)[1::2]
    print(f"x_quant_packed={x_quant_packed}")
    #print(x_quant_packed.size())
    #print(x_quant_packed.storage().nbytes())