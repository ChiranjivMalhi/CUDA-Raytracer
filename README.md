# CUDA Raytracer

A GPU-accelerated ray tracer built using **C++ and CUDA**, exploring parallel ray tracing, ray-object intersections, material interactions, and GPU memory management.

The project focuses on moving the computationally intensive ray-tracing workload from the CPU to the GPU using CUDA's parallel execution model.

## Demo


https://github.com/user-attachments/assets/d2a1dfbd-4355-4ae9-9364-7da3f63dde2d


## Features

* GPU-accelerated ray tracing using **CUDA**
* Parallel per-pixel ray generation
* Ray-sphere intersection
* Multiple ray bounces
* Lambertian diffuse materials
* Metallic materials with adjustable fuzziness
* Dielectric / glass materials
* Reflection and refraction
* Schlick's approximation for dielectric reflectance
* CUDA device memory management
* CPU-built median-split BVH, uploaded and traversed on the GPU
* Three side-by-side model instances with diffuse, metal, and glass materials
* Progressive sample accumulation across frames
* GPU-based image generation
* Tone mapping and gamma correction
* OpenGL/GLFW-based display

## Rendering

Each pixel is processed by a CUDA thread. The thread generates a camera ray and traces it through the scene, determining the closest object intersection and evaluating the material response.

Depending on the material, the ray can be scattered, reflected, or refracted. This process is repeated for multiple bounces until the ray terminates or reaches the maximum allowed depth.

The resulting color is written to a GPU image buffer and processed for display before being presented through OpenGL.

For OBJ scenes, the CPU loads vertex positions and faces, triangulates polygon faces, normalizes the model, and builds a BVH. The GPU traverses the BVH to find triangle intersections, alongside the ground sphere. Model mode displays three copies with diffuse, metal, and dielectric materials. The rendered image is tone-mapped on the GPU, copied back to the CPU, and displayed through OpenGL.

### Final Render (128 Samples)

<img width="956" height="525" alt="image" src="https://github.com/user-attachments/assets/f51904ec-6a86-44e3-953d-c6ce021655ef" />


<img width="957" height="535" alt="image" src="https://github.com/user-attachments/assets/aa16653a-e74e-4ae0-8274-5d645254bba7" />

---
## CPU vs GPU

The project was initially implemented as a **CPU ray tracer** and was then ported to **CUDA**, moving the computationally intensive ray-tracing workload to the GPU.

The CUDA implementation parallelizes the rendering workload at the pixel level, allowing thousands of CUDA threads to trace rays concurrently.

### Performance Comparison

| Metric         |                    CPU |               CUDA GPU |
| -------------- | ---------------------: | ---------------------: |
| Rendering Time |      **60 secs** |      **1secs** |
|Samples         | **4**| **4**|
| Resolution     | **960x540** | **960x540** |
| Max Ray Depth  |            **5** |            **5** |
| Speedup        |                 **1×** |         **60×** |

### Parallelization

The CPU implementation processes the image using conventional CPU execution, while the CUDA version maps pixel computation onto CUDA threads.

```text
CPU
 └── Pixel 1 → Ray Trace
 └── Pixel 2 → Ray Trace
 └── Pixel 3 → Ray Trace
 └── ...

CUDA GPU
 ├── Thread 1 → Pixel 1 → Ray Trace
 ├── Thread 2 → Pixel 2 → Ray Trace
 ├── Thread 3 → Pixel 3 → Ray Trace
 ├── Thread 4 → Pixel 4 → Ray Trace
 └── ...
```

The port preserves the core ray-tracing behavior of the CPU implementation while exploiting the GPU's massive parallelism.

## CUDA Implementation

### Parallel Ray Tracing

The image is divided into pixels, with CUDA threads processing pixels in parallel.

```text
Image
  ↓
CUDA Grid
  ↓
CUDA Blocks
  ↓
CUDA Threads
  ↓
Individual Pixels
```

This allows thousands of ray-tracing operations to execute concurrently on the GPU.

### GPU Memory

Scene data and rendering buffers are allocated in device memory and accessed directly by CUDA kernels.

The renderer manages GPU memory for:

* Scene objects
* Materials
* Camera data
* Ray-tracing output
* Intermediate rendering data

### Material Scattering

Each material defines how an incoming ray interacts with a surface:

* **Lambertian** — diffuse scattering
* **Metal** — reflection with configurable surface fuzziness
* **Dielectric** — reflection and refraction

For dielectric materials, **Schlick's approximation** is used to estimate the probability of reflection.


---

## Project Structure

```text
CUDA-Raytracer/
│
├── CUDA Raytracer/
│   ├── Raytracer.cpp
│   ├── cuda_kernels.cu
│   ├── cuda_kernels.cuh
│   ├── memory.cu
│   ├── memory.cuh
│   ├── material.cuh
│   ├── material_types.h
│   ├── camera.h
│   ├── ray.h
│   ├── sphere.h
│   ├── hittable.h
│   ├── hittable_list.h
│   ├── vec3.h
│   ├── color.h
│   └── constants.h
│
└── CUDA Raytracer.slnx
```

### Main Components

**`cuda_kernels.cu`**
Contains the CUDA kernels responsible for GPU ray tracing and image generation.

**`memory.cu`**
Handles allocation, initialization, and management of GPU-side rendering data.

**`material.cuh`**
Defines material behavior including diffuse scattering, reflection, and refraction.

**`ray.h`**
Defines the ray representation used throughout the renderer.

**`sphere.h` / `hittable.h`**
Define scene geometry and ray-object intersection functionality.

**`camera.h`**
Defines the camera and ray generation.

**`Raytracer.cpp`**
Handles application setup, scene construction, rendering configuration, and display.

---

## Technologies

**C++ · CUDA · OpenGL · GLFW · Visual Studio**

---

## Setup

### Requirements

* Windows
* Visual Studio
* NVIDIA GPU
* CUDA Toolkit
* CUDA-compatible compiler/toolchain

### Running

Clone the repository:

```bash
git clone https://github.com/ChiranjivMalhi/CUDA-Raytracer.git
```

Open `CUDA Raytracer.slnx` in Visual Studio, build the project, and run the application.

---

## Future Improvements

* **CUDA Optimization** — Optimize memory access patterns, thread occupancy, and kernel execution.
* **GPU Profiling** — Use NVIDIA Nsight tools to identify and optimize CPU/GPU bottlenecks.
* **CUDA/OpenGL Interoperability** — Reduce CPU-GPU data transfers by directly sharing CUDA output buffers with OpenGL.
* **Denoising** — Add spatial or temporal denoising to produce cleaner images with fewer samples.
* **Depth of Field & Motion Blur** — Extend camera ray generation to support additional physically-based effects.

---

## References

* **Ray Tracing in One Weekend** — Peter Shirley
  https://raytracing.github.io/books/RayTracingInOneWeekend.html

* **CUDA C++ Programming Guide** — NVIDIA
  https://docs.nvidia.com/cuda/cuda-c-programming-guide/

* **CUDA C++ Best Practices Guide** — NVIDIA
  https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/

* **CUDA Toolkit Documentation** — NVIDIA
  https://docs.nvidia.com/cuda/

* **Learn OpenGL** — Joey de Vries
  https://learnopengl.com/

---

## Author

**Chiranjiv Malhi**

https://github.com/ChiranjivMalhi
