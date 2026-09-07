#include "cuda_kernels.cuh"
#include "memory.cuh"
#include "material.cuh"
#include <curand_kernel.h>
#include <cuda_runtime.h>

__global__ void init_curand_kernel(curandState* states, unsigned long long seed, int num_threads) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_threads) {
        curand_init(seed + idx, 0, 0, &states[idx]);
    }
}
__global__ void ray_tracing_kernel(
    float4* accumulation,
    const GPUSphere* spheres,
    int num_spheres,
    const material* materials,
    const CameraGPU* camera,
    curandState* rand_states,
    int frame) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_pixels = camera->image_width * camera->image_height;

    if (idx >= total_pixels) return;

    int pixel_x = idx % camera->image_width;
    int pixel_y = idx / camera->image_width;

    curandState local_state = rand_states[idx];

    double u_offset = random_double(&local_state) - 0.5;
    double v_offset = random_double(&local_state) - 0.5;

    point3 pixel_sample = camera->pixel00_loc
        + ((pixel_x + u_offset) * camera->pixel_delta_u)
        + ((pixel_y + v_offset) * camera->pixel_delta_v);

    point3 ray_origin = camera->lookfrom;
    vec3 ray_direction = unit_vector(pixel_sample - ray_origin);

    ray r(ray_origin, ray_direction);

    color pixel_color(0, 0, 0);
    vec3 throughput(1.0, 1.0, 1.0);

    for (int bounce = 0; bounce < camera->max_depth; bounce++) {
        hit_record rec;
        bool hit = false;
        double closest_t = 1e30;
        int hit_index = -1;

        for (int i = 0; i < num_spheres; i++) {
            const GPUSphere& sphere = spheres[i];

            vec3 oc = r.origin() - sphere.center;
            double a = dot(r.direction(), r.direction());
            double h = dot(r.direction(), oc);
            double c = dot(oc, oc) - sphere.radius * sphere.radius;
            double discriminant = h * h - a * c;

            if (discriminant >= 0.0) {
                double sqrt_d = sqrt(discriminant);
                double root = (-h - sqrt_d) / a;

              
                if (root <= 0.001 || root >= closest_t) {
                    root = (-h + sqrt_d) / a;
                }

                if (root > 0.001 && root < closest_t) {
                    closest_t = root;
                    hit = true;
                    hit_index = i;
                }
            }
        }

   
        if (hit) {
            const GPUSphere& sphere = spheres[hit_index];
            rec.t = closest_t;
            rec.p = r.at(closest_t);
            vec3 outward_normal = (rec.p - sphere.center) / sphere.radius;
            rec.front_face = dot(r.direction(), outward_normal) < 0.0;
            rec.normal = rec.front_face ? outward_normal : -outward_normal;
            rec.mat = &materials[sphere.material_id];

            color attenuation;
            ray scattered;
            if (scatter(*rec.mat, r, rec, attenuation, scattered, &local_state)) {
                throughput = throughput * attenuation;
                r = scattered;
            }
            else {
                break;
            }
        }
        else {
            vec3 unit_direction = unit_vector(r.direction());
            double t = 0.5 * (unit_direction.y() + 1.0);
            color sky = (1.0 - t) * color(1.0, 1.0, 1.0) + t * color(0.5, 0.7, 1.0);
            pixel_color = throughput * sky;
            break;
        }
    }

    accumulation[idx].x += pixel_color.x();
    accumulation[idx].y += pixel_color.y();
    accumulation[idx].z += pixel_color.z();
    accumulation[idx].w += 1.0f;

    rand_states[idx] = local_state;
}

__global__ void tone_mapping_kernel(
    uint8_t* output_image,
    const float4* accumulation,
    int sample_count,
    int width,
    int height) {
    
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    int idx = y * width + x;
    float4 acc = accumulation[idx];
    
    float3 color_f = make_float3(
        acc.x / sample_count,
        acc.y / sample_count,
        acc.z / sample_count
    );
    
    float gamma = 1.0f / 2.2f;
    color_f.x = powf(fmaxf(color_f.x, 0.0f), gamma);
    color_f.y = powf(fmaxf(color_f.y, 0.0f), gamma);
    color_f.z = powf(fmaxf(color_f.z, 0.0f), gamma);
    
    uint8_t r = (uint8_t)(fminf(fmaxf(color_f.x, 0.0f), 1.0f) * 255.0f);
    uint8_t g = (uint8_t)(fminf(fmaxf(color_f.y, 0.0f), 1.0f) * 255.0f);
    uint8_t b = (uint8_t)(fminf(fmaxf(color_f.z, 0.0f), 1.0f) * 255.0f);
    
    int pixel_offset = idx * 3;
    output_image[pixel_offset] = r;
    output_image[pixel_offset + 1] = g;
    output_image[pixel_offset + 2] = b;
}
