#include "memory.cuh"
#include "cuda_kernels.cuh"
#include <cuda_runtime.h>
#include <iostream>

GPUScene::GPUScene() 
    : d_spheres(nullptr), d_triangles(nullptr), d_bvh_nodes(nullptr),
      d_triangle_indices(nullptr), d_materials(nullptr), num_spheres(0),
      num_triangles(0), num_bvh_nodes(0), d_accumulation(nullptr),
      d_output_image(nullptr), d_rand_states(nullptr), d_camera(nullptr) {}

GPUScene::~GPUScene() {
    deallocate();
}

void GPUScene::allocate(int width, int height, int num_spheres_, int num_materials,
                        int num_triangles_, int num_bvh_nodes_) {
    num_spheres = num_spheres_;
    num_triangles = num_triangles_;
    num_bvh_nodes = num_bvh_nodes_;
    int num_pixels = width * height;
    
    if (num_spheres > 0) cudaMalloc(&d_spheres, num_spheres * sizeof(GPUSphere));
    if (num_materials > 0) cudaMalloc(&d_materials, num_materials * sizeof(material));
    if (num_triangles > 0) cudaMalloc(&d_triangles, num_triangles * sizeof(GPUTriangle));
    if (num_bvh_nodes > 0) cudaMalloc(&d_bvh_nodes, num_bvh_nodes * sizeof(GPUBvhNode));
    if (num_triangles > 0) cudaMalloc(&d_triangle_indices, num_triangles * sizeof(int));
    
    cudaMalloc(&d_accumulation, num_pixels * sizeof(float4));
    cudaMemset(d_accumulation, 0, num_pixels * sizeof(float4));
    
    cudaMalloc(&d_output_image, num_pixels * 3 * sizeof(uint8_t));

    cudaMalloc(&d_rand_states, num_pixels * sizeof(curandState));

    cudaMalloc(&d_camera, sizeof(CameraGPU));
}

void GPUScene::deallocate() {
    if (d_spheres) cudaFree(d_spheres);
    if (d_triangles) cudaFree(d_triangles);
    if (d_bvh_nodes) cudaFree(d_bvh_nodes);
    if (d_triangle_indices) cudaFree(d_triangle_indices);
    if (d_materials) cudaFree(d_materials);
    if (d_accumulation) cudaFree(d_accumulation);
    if (d_output_image) cudaFree(d_output_image);
    if (d_rand_states) cudaFree(d_rand_states);
    if (d_camera) cudaFree(d_camera);
}

void GPUScene::transfer_scene_data(const std::vector<GPUSphere>& cpu_spheres, 
                                   const std::vector<material>& cpu_materials,
                                   const std::vector<GPUTriangle>& cpu_triangles,
                                   const std::vector<GPUBvhNode>& cpu_bvh_nodes,
                                   const std::vector<int>& cpu_triangle_indices) {
    if (!cpu_spheres.empty()) cudaMemcpy(d_spheres, cpu_spheres.data(), cpu_spheres.size() * sizeof(GPUSphere), cudaMemcpyHostToDevice);
    if (!cpu_materials.empty()) cudaMemcpy(d_materials, cpu_materials.data(), cpu_materials.size() * sizeof(material), cudaMemcpyHostToDevice);
    if (!cpu_triangles.empty()) cudaMemcpy(d_triangles, cpu_triangles.data(), cpu_triangles.size() * sizeof(GPUTriangle), cudaMemcpyHostToDevice);
    if (!cpu_bvh_nodes.empty()) cudaMemcpy(d_bvh_nodes, cpu_bvh_nodes.data(), cpu_bvh_nodes.size() * sizeof(GPUBvhNode), cudaMemcpyHostToDevice);
    if (!cpu_triangle_indices.empty()) cudaMemcpy(d_triangle_indices, cpu_triangle_indices.data(), cpu_triangle_indices.size() * sizeof(int), cudaMemcpyHostToDevice);
}

void GPUScene::get_output_image(std::vector<uint8_t>& cpu_output) {
    int num_bytes = cpu_output.size();
    cudaMemcpy(cpu_output.data(), d_output_image, num_bytes, cudaMemcpyDeviceToHost);
}
