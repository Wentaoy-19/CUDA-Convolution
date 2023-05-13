#if 1
#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"
#include "cuda_fp16.h"


#define TILE_BATCH 1
#define BLOCK_WIDTH 16


__constant__ half mask_c[7*7*16*4];
static half* fp16_input;
static half* fp16_output;
static half* fp16_mask;

// Half precision ref: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__INTRINSIC__HALF.html#group__CUDA__MATH__INTRINSIC__HALF
 
__global__ void float_fp16(const float* input,half* output,int size)
{
    int idx = blockDim.x*blockIdx.x + threadIdx.x;
    if(idx < size)
        output[idx] = __float2half(input[idx]);
}

__global__ void fp16_float(half* input, float* output, int size)
{
    int idx = blockDim.x*blockIdx.x + threadIdx.x;
    if(idx < size)
        output[idx] = __half2float(input[idx]);
}

__global__ void conv_forward_kernel_fp16(half *output, const half*  input, const half*  mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{

    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;

    #define out_4d(i3, i2, i1, i0) output[(i3) * (Map_out * Height_out * Width_out) + (i2) * (Height_out * Width_out) + (i1) * (Width_out) + i0]
    #define in_4d(i3, i2, i1, i0) input[(i3) * (Channel * Height * Width) + (i2) * (Height * Width) + (i1) * (Width) + i0]
    #define mask_4d(i3, i2, i1, i0) mask[(i3) * (Channel * K * K) + (i2) * (K * K) + (i1) * (K) + i0]
    #define mask_4d_c(i3, i2, i1, i0) mask_c[(i3) * (Channel * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    // Insert your GPU convolution kernel code here

    int h_grid = ceil(1.0*Height_out/BLOCK_WIDTH);
    int w_grid = ceil(1.0*Width_out/BLOCK_WIDTH);

    int m = blockIdx.x;
    int h = (blockIdx.y/w_grid)*BLOCK_WIDTH + threadIdx.y;
    int w = (blockIdx.y%w_grid)*BLOCK_WIDTH + threadIdx.x;
    int b = blockIdx.z;

    if(h<Height_out && w<Width_out)
    {
        half2 acc_2;


        for(int c=0;c<Channel;c++)
        {
            for(int p=0;p<K;p++)
            {
                half2 fp_in[4] = {half2(in_4d(b,c,h+p,w),in_4d(b,c,h+p,w+1)),
                                  half2(in_4d(b,c,h+p,w+2),in_4d(b,c,h+p,w+3)),
                                  half2(in_4d(b,c,h+p,w+4),in_4d(b,c,h+p,w+5)),
                                  half2(in_4d(b,c,h+p,w+6),0)};
                half2 fp_mask[4] = {half2(mask_4d(m,c,p,0),mask_4d(m,c,p,1)),
                                    half2(mask_4d(m,c,p,2),mask_4d(m,c,p,3)),
                                    half2(mask_4d(m,c,p,4),mask_4d(m,c,p,5)),
                                    half2(mask_4d(m,c,p,6),0)};
                for(int i=0;i<4;i++)
                {
                    acc_2 = __hadd2(acc_2,__hmul2(fp_in[i],fp_mask[i]));
                }


                // half2 a_in;
                // half2 b_mask;
                // #pragma unroll 4
                // for(int q=0;q<4;q++)
                // {
                //     b_mask.x = mask_4d(m,c,p,2*q);
                //     a_in.x = in_4d(b,c,h+p,w+2*q);
                //     if(2*q+1 < K)
                //     {
                //         b_mask.y = mask_4d(m,c,p,2*q+1);
                //         a_in.y = in_4d(b,c,h+p,w+2*q+1);
                //     }
                //     else
                //     {
                //         b_mask.y = 0;
                //         a_in.y = 0;               
                //     }
                //     acc_2 = __hadd2(acc_2,__hmul2(a_in,b_mask));               
                // }

                // for(int q=0;q<K;q+=2)
                // {
                //     b_mask.x = mask_4d(m,c,p,q);
                //     a_in.x = in_4d(b,c,h+p,w+q);
                //     if(q+1 < K)
                //     {
                //         b_mask.y = mask_4d(m,c,p,q+1);
                //         a_in.y = in_4d(b,c,h+p,w+q+1);
                //     }
                //     else
                //     {
                //         b_mask.y = 0;
                //         a_in.y = 0;               
                //     }
                //     acc_2 = __hadd2(acc_2,__hmul2(a_in,b_mask));
                // }
            }
        }
        out_4d(b,m,h,w) = acc_2.y;
    }

    #undef out_4d
    #undef in_4d
    #undef mask_4d
    #undef mask_4d_c
}


	
__host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_output, const float *host_input, const float *host_mask, float **device_output_ptr, float **device_input_ptr, float **device_mask_ptr, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;
    cudaMalloc((void**)device_input_ptr,sizeof(float)*Batch*Channel*Height*Width);
    cudaMalloc((void**)device_mask_ptr,sizeof(float)*K*K*Channel*Map_out);
    cudaMalloc((void**)device_output_ptr,sizeof(float)*Height_out*Width_out*Map_out*Batch);    
    cudaMemcpy(*device_input_ptr,host_input,sizeof(float)*Batch*Channel*Height*Width,cudaMemcpyHostToDevice);
    cudaMemcpy(*device_mask_ptr,host_mask,sizeof(float)*K*K*Channel*Map_out,cudaMemcpyHostToDevice);
    // cudaMemcpyToSymbol(mask_c,host_mask,sizeof(float)*Map_out*Channel*K*K);
    



    // convert to fp16
    cudaMalloc((void**)&fp16_input,sizeof(half)*Batch*Channel*Height*Width);
    cudaMalloc((void**)&fp16_mask,sizeof(half)*K*K*Map_out*Channel);
    cudaMalloc((void**)&fp16_output,sizeof(half)*Height_out*Width_out*Batch*Map_out);
    int h_grid = ceil(1.0*Height_out/BLOCK_WIDTH);
    int w_grid = ceil(1.0*Width_out/BLOCK_WIDTH);

    dim3 dim_block0(BLOCK_WIDTH*BLOCK_WIDTH,1,1);
    dim3 dim_grid0(ceil(1.0*Batch*Channel*Height*Width/(BLOCK_WIDTH*BLOCK_WIDTH)),1,1);
    float_fp16<<<dim_grid0,dim_block0>>>(*device_input_ptr,fp16_input,Height*Width*Channel*Batch);
    cudaDeviceSynchronize();

    dim3 dim_block1(K*K,1,1);
    dim3 dim_grid1(ceil(1.0*K*K*Map_out*Channel/(K*K)),1,1);
    float_fp16<<<dim_grid1,dim_block1>>>(*device_mask_ptr,fp16_mask,K*K*Map_out*Channel);
    cudaDeviceSynchronize();

    // cudaMemcpyToSymbol(mask_c,fp16_mask,sizeof(half)*Map_out*Channel*K*K);



}


__host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{

    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;
    int h_grid = ceil(1.0*Height_out/BLOCK_WIDTH);
    int w_grid = ceil(1.0*Width_out/BLOCK_WIDTH);

    dim3 dim_block(BLOCK_WIDTH,BLOCK_WIDTH,1);
    dim3 dim_grid(Map_out,h_grid*w_grid,Batch);
    conv_forward_kernel_fp16<<<dim_grid,dim_block>>>(fp16_output, fp16_input, fp16_mask, Batch,Map_out, Channel, Height,Width,  K);
    // cudaDeviceSynchronize();

}


__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // Copy the output back to host
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;

    dim3 dim_block2(BLOCK_WIDTH*BLOCK_WIDTH,1,1);
    dim3 dim_grid2(ceil(1.0*Height_out*Width_out*Batch*Map_out/(BLOCK_WIDTH*BLOCK_WIDTH)),1,1);
    fp16_float<<<dim_grid2,dim_block2>>>(fp16_output,device_output,Height_out*Width_out*Batch*Map_out);
    cudaDeviceSynchronize();

    cudaMemcpy(host_output,device_output,sizeof(float)*Batch*Map_out*Height_out*Width_out,cudaMemcpyDeviceToHost);
    // Free device memory
    cudaFree(device_input);
    cudaFree(device_output);
    cudaFree(device_mask);
}


__host__ void GPUInterface::get_device_properties()
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    for(int dev = 0; dev < deviceCount; dev++)
    {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        std::cout<<"Device "<<dev<<" name: "<<deviceProp.name<<std::endl;
        std::cout<<"Computational capabilities: "<<deviceProp.major<<"."<<deviceProp.minor<<std::endl;
        std::cout<<"Max Global memory size: "<<deviceProp.totalGlobalMem<<std::endl;
        std::cout<<"Max Constant memory size: "<<deviceProp.totalConstMem<<std::endl;
        std::cout<<"Max Shared memory size per block: "<<deviceProp.sharedMemPerBlock<<std::endl;
        std::cout<<"Max threads per block: "<<deviceProp.maxThreadsPerBlock<<std::endl;
        std::cout<<"Max block dimensions: "<<deviceProp.maxThreadsDim[0]<<" x, "<<deviceProp.maxThreadsDim[1]<<" y, "<<deviceProp.maxThreadsDim[2]<<" z"<<std::endl;
        std::cout<<"Max grid dimensions: "<<deviceProp.maxGridSize[0]<<" x, "<<deviceProp.maxGridSize[1]<<" y, "<<deviceProp.maxGridSize[2]<<" z"<<std::endl;
        std::cout<<"Warp Size: "<<deviceProp.warpSize<<std::endl;
    }
}
#endif