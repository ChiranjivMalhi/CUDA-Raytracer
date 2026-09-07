#ifndef CUDA_KERNELS_CUH
#define CUDA_KERNELS_CUH

#include "vec3.h"
#include "ray.h"
#include "hittable.h"
#include "material_types.h"
#include <curand_kernel.h>
#include <stdint.h>

struct GPUSphere {
    point3 center;
    double radius;
    int material_id;
};

struct CameraGPU {
    point3 lookfrom;
    point3 pixel00_loc;
    vec3 pixel_delta_u;
    vec3 pixel_delta_v;
    int image_width;
    int image_height;
    int max_depth;
};

__global__ void init_curand_kernel(curandState* states, unsigned long long seed, int num_threads);

__global__ void ray_tracing_kernel(
    float4* accumulation,
    const GPUSphere* spheres,
    int num_spheres,
    const material* materials,
    const CameraGPU* camera,
    curandState* rand_states,
    int frame);

__global__ void tone_mapping_kernel(
    uint8_t* output_image,
    const float4* accumulation,
    int sample_count,
    int width,
    int height);

#endif