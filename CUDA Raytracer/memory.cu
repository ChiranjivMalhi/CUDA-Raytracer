#include "memory.cuh"
#include "cuda_kernels.cuh"
#include <cuda_runtime.h>
#include <iostream>

GPUScene::GPUScene() 
    : d_spheres(nullptr), d_materials(nullptr), d_accumulation(nullptr),
      d_output_image(nullptr), d_rand_states(nullptr), d_camera(nullptr),
      num_spheres(0) {}

GPUScene::~GPUScene() {
    deallocate();
}

void GPUScene::allocate(int width, int height, int num_spheres_) {
    num_spheres = num_spheres_;
    int num_pixels = width * height;
    
    cudaMalloc(&d_spheres, num_spheres * sizeof(GPUSphere));
    cudaMalloc(&d_materials, num_spheres * sizeof(material));
    
    cudaMalloc(&d_accumulation, num_pixels * sizeof(float4));
    cudaMemset(d_accumulation, 0, num_pixels * sizeof(float4));
    
    cudaMalloc(&d_output_image, num_pixels * 3 * sizeof(uint8_t));

    cudaMalloc(&d_rand_states, num_pixels * sizeof(curandState));

    cudaMalloc(&d_camera, sizeof(CameraGPU));
}

void GPUScene::deallocate() {
    if (d_spheres) cudaFree(d_spheres);
    if (d_materials) cudaFree(d_materials);
    if (d_accumulation) cudaFree(d_accumulation);
    if (d_output_image) cudaFree(d_output_image);
    if (d_rand_states) cudaFree(d_rand_states);
    if (d_camera) cudaFree(d_camera);
}

void GPUScene::transfer_scene_data(const std::vector<GPUSphere>& cpu_spheres, 
                                   const std::vector<material>& cpu_materials) {
    cudaMemcpy(d_spheres, cpu_spheres.data(), cpu_spheres.size() * sizeof(GPUSphere), 
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_materials, cpu_materials.data(), cpu_materials.size() * sizeof(material), 
               cudaMemcpyHostToDevice);
}

void GPUScene::get_output_image(std::vector<uint8_t>& cpu_output) {
    int num_bytes = cpu_output.size();
    cudaMemcpy(cpu_output.data(), d_output_image, num_bytes, cudaMemcpyDeviceToHost);
}
