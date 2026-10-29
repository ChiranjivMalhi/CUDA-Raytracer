#include "cuda_kernels.cuh"
#include "memory.cuh"
#include "material.cuh"
#include <curand_kernel.h>
#include <cuda_runtime.h>

__device__ bool hit_aabb(const ray& r, const GPUBvhNode& node, double max_t) {
    double t_min = 0.001;
    double t_max = max_t;
    for (int axis = 0; axis < 3; ++axis) {
        const double origin = r.origin()[axis];
        const double direction = r.direction()[axis];
        const double lo = node.bounds_min[axis];
        const double hi = node.bounds_max[axis];
        if (fabs(direction) < 1e-15) {
            if (origin < lo || origin > hi) return false;
            continue;
        }
        double t0 = (lo - origin) / direction;
        double t1 = (hi - origin) / direction;
        if (t0 > t1) { const double temp = t0; t0 = t1; t1 = temp; }
        t_min = fmax(t_min, t0);
        t_max = fmin(t_max, t1);
        if (t_max < t_min) return false;
    }
    return true;
}

__device__ bool hit_triangle(const ray& r, const GPUTriangle& tri, double max_t,
                             double& t, vec3& normal) {
    const vec3 edge1 = tri.v1 - tri.v0;
    const vec3 edge2 = tri.v2 - tri.v0;
    const vec3 p = cross(r.direction(), edge2);
    const double determinant = dot(edge1, p);
    if (fabs(determinant) < 1e-12) return false;
    const double inverse_determinant = 1.0 / determinant;
    const vec3 from_vertex = r.origin() - tri.v0;
    const double u = dot(from_vertex, p) * inverse_determinant;
    if (u < 0.0 || u > 1.0) return false;
    const vec3 q = cross(from_vertex, edge1);
    const double v = dot(r.direction(), q) * inverse_determinant;
    if (v < 0.0 || u + v > 1.0) return false;
    const double candidate_t = dot(edge2, q) * inverse_determinant;
    if (candidate_t <= 0.001 || candidate_t >= max_t) return false;
    const vec3 unnormalized_normal = cross(edge1, edge2);
    const double normal_length = unnormalized_normal.length();
    if (normal_length < 1e-15) return false;
    t = candidate_t;
    normal = unnormalized_normal / normal_length;
    return true;
}

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
    const GPUTriangle* triangles,
    const GPUBvhNode* bvh_nodes,
    int num_bvh_nodes,
    const int* triangle_indices,
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
        }

        if (num_bvh_nodes > 0) {
            int stack[64];
            int stack_size = 0;
            stack[stack_size++] = 0;
            while (stack_size > 0) {
                const int node_index = stack[--stack_size];
                const GPUBvhNode& node = bvh_nodes[node_index];
                if (!hit_aabb(r, node, closest_t)) continue;

                if (node.count > 0) {
                    for (int j = 0; j < node.count; ++j) {
                        const GPUTriangle& tri = triangles[triangle_indices[node.first + j]];
                        double triangle_t;
                        vec3 outward_normal;
                        if (hit_triangle(r, tri, closest_t, triangle_t, outward_normal)) {
                            closest_t = triangle_t;
                            hit = true;
                            rec.t = triangle_t;
                            rec.p = r.at(triangle_t);
                            rec.front_face = dot(r.direction(), outward_normal) < 0.0;
                            rec.normal = rec.front_face ? outward_normal : -outward_normal;
                            rec.mat = &materials[tri.material_id];
                        }
                    }
                } else if (stack_size + 2 <= 64) {
                    stack[stack_size++] = node.right;
                    stack[stack_size++] = node.left;
                }
            }
        }

        if (hit) {
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
