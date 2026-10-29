#ifndef MEMORY_CUH
#define MEMORY_CUH

#include "cuda_kernels.cuh"
#include <vector>
#include <cuda_runtime.h>

class GPUScene {
public:
    GPUSphere* d_spheres;
    GPUTriangle* d_triangles;
    GPUBvhNode* d_bvh_nodes;
    int* d_triangle_indices;
    material* d_materials;
    int num_spheres;
    int num_triangles;
    int num_bvh_nodes;
    
    float4* d_accumulation;
    uint8_t* d_output_image;
    curandState* d_rand_states;
    
    CameraGPU* d_camera;
    
    GPUScene();
    ~GPUScene();
    
    void allocate(int width, int height, int num_spheres, int num_materials,
                  int num_triangles, int num_bvh_nodes);
    void deallocate();
    void transfer_scene_data(const std::vector<GPUSphere>& cpu_spheres,
                             const std::vector<material>& cpu_materials,
                             const std::vector<GPUTriangle>& cpu_triangles,
                             const std::vector<GPUBvhNode>& cpu_bvh_nodes,
                             const std::vector<int>& cpu_triangle_indices);
    void get_output_image(std::vector<uint8_t>& cpu_output);
};

#endif
