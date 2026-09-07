#ifndef MATERIAL_TYPES_H
#define MATERIAL_TYPES_H

#include "vec3.h"

enum class material_type {
    lambertian,
    metal,
    dielectric
};

struct material {
    material_type type;
    color albedo;
    double fuzz;              
    double refraction_index;  

    __host__ __device__ material() 
        : type(material_type::lambertian), albedo(color(0, 0, 0)), fuzz(0.0), refraction_index(0.0) {}

    __host__ __device__ material(const color& a, material_type t) 
        : type(t), albedo(a), fuzz(0.0), refraction_index(0.0) {}

    __host__ __device__ material(const color& a, double f) 
        : type(material_type::metal), albedo(a), fuzz(f), refraction_index(0.0) {}

    __host__ __device__ material(double ri) 
        : type(material_type::dielectric), albedo(color(1, 1, 1)), fuzz(0.0), refraction_index(ri) {}
};

#endif
