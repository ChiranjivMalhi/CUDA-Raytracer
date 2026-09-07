#ifndef CUDA_MATERIALS_CUH
#define CUDA_MATERIALS_CUH

#include <curand_kernel.h>

#include "vec3.h"
#include "ray.h"
#include "hittable.h"
#include "material_types.h"


__device__ inline double random_double(curandState* rng) {
    return curand_uniform_double(rng);
}

__device__ inline double random_double(curandState* rng, double min, double max) {
    return min + (max - min) * random_double(rng);
}

__device__ inline vec3 random_vec3(curandState* rng) {
    return vec3(random_double(rng), random_double(rng), random_double(rng));
}

__device__ inline vec3 random_vec3(curandState* rng, double min, double max) {
    return vec3(
        random_double(rng, min, max),
        random_double(rng, min, max),
        random_double(rng, min, max));
}

__device__ inline vec3 random_unit_vector(curandState* rng) {
    while (true) {
        vec3 p = random_vec3(rng, -1.0, 1.0);
        double lensq = p.length_squared();
        if (1e-160 < lensq && lensq <= 1.0)
            return p / sqrt(lensq);
    }
}

__device__ inline double reflectance(double cosine, double refraction_index) {
    // Schlick's approximation for reflectance.
    auto r0 = (1 - refraction_index) / (1 + refraction_index);
    r0 = r0 * r0;
    return r0 + (1 - r0) * pow((1 - cosine), 5);
}

__device__ inline bool scatter(
    const material& mat,
    const ray& r_in,
    const hit_record& rec,
    color& attenuation,
    ray& scattered,
    curandState* rng) {
    switch (mat.type) {
    case material_type::lambertian: {
        vec3 scatter_direction = rec.normal + random_unit_vector(rng);

        if (scatter_direction.near_zero())
            scatter_direction = rec.normal;

        scattered = ray(rec.p, scatter_direction);
        attenuation = mat.albedo;
        return true;
    }

    case material_type::metal: {
        vec3 reflected = reflect(r_in.direction(), rec.normal);
        reflected = unit_vector(reflected) + (mat.fuzz * random_unit_vector(rng));
        scattered = ray(rec.p, reflected);
        attenuation = mat.albedo;
        return true;
    }

    case material_type::dielectric: {
        attenuation = color(1.0, 1.0, 1.0);
        double ri = rec.front_face ? (1.0 / mat.refraction_index) : mat.refraction_index;

        vec3 unit_direction = unit_vector(r_in.direction());
        double cos_theta = fmin(dot(-unit_direction, rec.normal), 1.0);
        double sin_theta = sqrt(1.0 - cos_theta * cos_theta);

        bool cannot_refract = ri * sin_theta > 1.0;
        vec3 direction;

        if (cannot_refract || reflectance(cos_theta, ri) > random_double(rng))
            direction = reflect(unit_direction, rec.normal);
        else
            direction = refract(unit_direction, rec.normal, ri);

        scattered = ray(rec.p, direction);
        return true;
    }
    }

    return false;
}

#endif
