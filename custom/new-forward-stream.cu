#if 0
#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"

#define TILE_BATCH 1
#define BLOCK_WIDTH 16
#define N_STREAM 5

static cudaStream_t s_list[N_STREAM];
static float* d_in;
static float* d_out;


__constant__ float mask_c[7*7*16*4];

__global__ void conv_forward_kernel(float *output, const float *input, const float *mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{

    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;

    #define out_4d(i3, i2, i1, i0) output[(i3) * (Map_out * Height_out * Width_out) + (i2) * (Height_out * Width_out) + (i1) * (Width_out) + i0]
    #define in_4d(i3, i2, i1, i0) input[(i3) * (Channel * Height * Width) + (i2) * (Height * Width) + (i1) * (Width) + i0]
    #define mask_4d(i3, i2, i1, i0) mask[(i3) * (Channel * K * K) + (i2) * (K * K) + (i1) * (K) + i0]
    #define mask_4d_c(i3, i2, i1, i0) mask_c[(i3) * (Channel * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    int h_grid = ceil(1.0*Height_out/BLOCK_WIDTH);
    int w_grid = ceil(1.0*Width_out/BLOCK_WIDTH);

    int m = blockIdx.x;
    int h = (blockIdx.y/w_grid)*BLOCK_WIDTH + threadIdx.y;
    int w = (blockIdx.y%w_grid)*BLOCK_WIDTH + threadIdx.x;
    int b = blockIdx.z;


    if(h<Height_out && w<Width_out)
    {
        float acc = 0.0f;
        for(int c=0;c<Channel;c++)
        {
            for(int p=0;p<K;p++)
            {
                for(int q =0;q<K;q++)
                {
                    acc += in_4d(b,c,h+p,w+q)*mask_4d_c(m,c,p,q);
                }
            }
        }
        out_4d(b,m,h,w) = acc;
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

    // *device_input_ptr = (float*)malloc(sizeof(float)*Batch*Channel*Height*Width);
    // *device_output_ptr = (float*)malloc(sizeof(float)*Height_out*Width_out*Map_out*Batch) ;
    // memcpy(*device_input_ptr,host_input,sizeof(float)*Batch*Channel*Height*Width);
    // *device_input_ptr = (float*)host_input;
    // *device_output_ptr = (float*)host_output;

    cudaMemcpyToSymbol(mask_c,host_mask,sizeof(float)*Map_out*Channel*K*K);
    for(int i=0;i<N_STREAM;i++)
    {
        cudaStreamCreate(&s_list[i]);
    }
    cudaMalloc((void**)&d_in,sizeof(float)*Batch*Channel*Height*Width);
    cudaMalloc((void**)&d_out,sizeof(float)*Batch*Map_out*Height_out*Width_out);


    // const int tile_batch_size = ceil(1.0*Batch/N_STREAM);
    const int tile_batch_size = 50;
    int h_grid = ceil(1.0*Height_out/BLOCK_WIDTH);
    int w_grid = ceil(1.0*Width_out/BLOCK_WIDTH);
    dim3 dim_block(BLOCK_WIDTH,BLOCK_WIDTH,1);
    dim3 dim_grid(Map_out,h_grid*w_grid,tile_batch_size);


    for(int b=0;b<ceil(1.0*Batch/(N_STREAM*tile_batch_size));b++)
    {
        for(int i=0;i<N_STREAM;i++)
        {
            int offset_in = (i + b*N_STREAM)*tile_batch_size*Channel*Height*Width;
            cudaMemcpyAsync(d_in + offset_in, host_input+offset_in,sizeof(float)*tile_batch_size*Channel*Height*Width,cudaMemcpyHostToDevice,s_list[i]);
        }

        for(int i=0;i<N_STREAM;i++)
        {
            int offset_in = (i + b*N_STREAM)*tile_batch_size*Channel*Height*Width;
            int offset_out = (i + b*N_STREAM)*tile_batch_size*Map_out*Height_out*Width_out;
            conv_forward_kernel<<<dim_grid,dim_block,0,s_list[i]>>>(d_out + offset_out, d_in + offset_in, host_mask, tile_batch_size, Map_out, Channel, Height, Width, K);
        }

        for(int i=0;i<N_STREAM;i++)
        {
            int offset_out = (i + b*N_STREAM)*tile_batch_size*Map_out*Height_out*Width_out;
            cudaMemcpyAsync((float*)host_output+offset_out,d_out + offset_out,sizeof(float)*tile_batch_size*Map_out*Height_out*Width_out,cudaMemcpyDeviceToHost,s_list[i]);
        }
    }
    cudaFree(d_in);
    cudaFree(d_out);


}

// stream ref: https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/#asynchronous-transfers-and-overlapping-transfers-with-computation

__host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // const int Height_out = Height - K + 1;
    // const int Width_out = Width - K + 1;

    // // const int tile_batch_size = ceil(1.0*Batch/N_STREAM);
    // const int tile_batch_size = 50;
    // int h_grid = ceil(1.0*Height_out/BLOCK_WIDTH);
    // int w_grid = ceil(1.0*Width_out/BLOCK_WIDTH);
    // dim3 dim_block(BLOCK_WIDTH,BLOCK_WIDTH,1);
    // dim3 dim_grid(Map_out,h_grid*w_grid,tile_batch_size);


    // for(int b=0;b<ceil(1.0*Batch/(N_STREAM*tile_batch_size));b++)
    // {
    //     // int offset_in0 = ( b*N_STREAM)*tile_batch_size*Channel*Height*Width;
    //     // int offset_in1 = ( b*N_STREAM + 1)*tile_batch_size*Channel*Height*Width;
    //     // int offset_out0 = ( b*N_STREAM)*tile_batch_size*Map_out*Height_out*Width_out;
    //     // int offset_out1 = (1 + b*N_STREAM)*tile_batch_size*Map_out*Height_out*Width_out;
    //     // cudaMemcpyAsync(d_in + offset_in0, device_input+offset_in0,sizeof(float)*tile_batch_size*Channel*Height*Width,cudaMemcpyHostToDevice,s_list[0]);
    //     // cudaMemcpyAsync(d_in + offset_in1, device_input+offset_in1,sizeof(float)*tile_batch_size*Channel*Height*Width,cudaMemcpyHostToDevice,s_list[1]);
    //     // conv_forward_kernel<<<dim_grid,dim_block,0,s_list[0]>>>(d_out + offset_out0, d_in + offset_in0, device_mask, tile_batch_size, Map_out, Channel, Height, Width, K);
    //     // conv_forward_kernel<<<dim_grid,dim_block,0,s_list[1]>>>(d_out + offset_out1, d_in + offset_in1, device_mask, tile_batch_size, Map_out, Channel, Height, Width, K);
    //     // cudaMemcpyAsync(device_output+offset_out0,d_out + offset_out0,sizeof(float)*tile_batch_size*Map_out*Height_out*Width_out,cudaMemcpyDeviceToHost,s_list[0]);
    //     // cudaMemcpyAsync(device_output+offset_out1,d_out + offset_out1,sizeof(float)*tile_batch_size*Map_out*Height_out*Width_out,cudaMemcpyDeviceToHost,s_list[1]);
    //     for(int i=0;i<N_STREAM;i++)
    //     {
    //         int offset_in = (i + b*N_STREAM)*tile_batch_size*Channel*Height*Width;
    //         cudaMemcpyAsync(d_in + offset_in, device_input+offset_in,sizeof(float)*tile_batch_size*Channel*Height*Width,cudaMemcpyHostToDevice,s_list[i]);
    //     }

    //     for(int i=0;i<N_STREAM;i++)
    //     {
    //         int offset_in = (i + b*N_STREAM)*tile_batch_size*Channel*Height*Width;
    //         int offset_out = (i + b*N_STREAM)*tile_batch_size*Map_out*Height_out*Width_out;
    //         conv_forward_kernel<<<dim_grid,dim_block,0,s_list[i]>>>(d_out + offset_out, d_in + offset_in, device_mask, tile_batch_size, Map_out, Channel, Height, Width, K);
    //     }

    //     for(int i=0;i<N_STREAM;i++)
    //     {
    //         int offset_out = (i + b*N_STREAM)*tile_batch_size*Map_out*Height_out*Width_out;
    //         cudaMemcpyAsync(device_output+offset_out,d_out + offset_out,sizeof(float)*tile_batch_size*Map_out*Height_out*Width_out,cudaMemcpyDeviceToHost,s_list[i]);
    //     }



    //     // for(int i=0;i<N_STREAM;i++)
    //     // {
    //     //     int offset_in = (i + b*N_STREAM)*tile_batch_size*Channel*Height*Width;
    //     //     int offset_out = (i + b*N_STREAM)*tile_batch_size*Map_out*Height_out*Width_out;
    //     //     cudaMemcpyAsync(d_in + offset_in, device_input+offset_in,sizeof(float)*tile_batch_size*Channel*Height*Width,cudaMemcpyHostToDevice,s_list[i]);
    //     //     conv_forward_kernel<<<dim_grid,dim_block,0,s_list[i]>>>(d_out + offset_out, d_in + offset_in, device_mask, tile_batch_size, Map_out, Channel, Height, Width, K);
    //     //     cudaMemcpyAsync(device_output+offset_out,d_out + offset_out,sizeof(float)*tile_batch_size*Map_out*Height_out*Width_out,cudaMemcpyDeviceToHost,s_list[i]);
    //     // }
    // }


    // // for(int i=0;i<N_STREAM;i++)
    // // {
    // //     int offset_in = i*tile_batch_size*Channel*Height*Width;
    // //     int offset_out = i*tile_batch_size*Map_out*Height_out*Width_out;
    // //     cudaMemcpyAsync(d_in + offset_in, device_input+offset_in,sizeof(float)*tile_batch_size*Channel*Height*Width,cudaMemcpyHostToDevice,s_list[i]);
    // //     conv_forward_kernel<<<dim_grid,dim_block,0,s_list[i]>>>(d_out + offset_out, d_in + offset_in, device_mask, tile_batch_size, Map_out, Channel, Height, Width, K);
    // //     cudaMemcpyAsync(device_output+offset_out,d_out + offset_out,sizeof(float)*tile_batch_size*Map_out*Height_out*Width_out,cudaMemcpyDeviceToHost,s_list[i]);
    // // }

}


__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // Copy the output back to host
    // const int Height_out = Height - K + 1;
    // const int Width_out = Width - K + 1;
    for(int i=0;i<N_STREAM;i++)
    {
        cudaStreamDestroy(s_list[i]);
    }

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

