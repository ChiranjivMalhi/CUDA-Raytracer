# CUDA Raytracer

A GPU-accelerated ray tracer built using **C++ and CUDA**, exploring parallel ray tracing, ray-object intersections, material interactions, and GPU memory management.

The project focuses on moving the computationally intensive ray-tracing workload from the CPU to the GPU using CUDA's parallel execution model.

## Demo


https://github.com/user-attachments/assets/236e4fd6-4260-4615-9df1-d5236d697f75

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
* GPU-based image generation
* Tone mapping and gamma correction
* OpenGL/GLFW-based display

## Rendering

Each pixel is processed by a CUDA thread. The thread generates a camera ray and traces it through the scene, determining the closest object intersection and evaluating the material response.

Depending on the material, the ray can be scattered, reflected, or refracted. This process is repeated for multiple bounces until the ray terminates or reaches the maximum allowed depth.

The resulting color is written to a GPU image buffer and processed for display before being presented through OpenGL.

### Final Render (128 Samples)

<img width="953" height="537" alt="image" src="https://github.com/user-attachments/assets/c49c13c2-9370-4cf7-9713-a84a436e23f0" />

<img width="951" height="532" alt="Screenshot 2026-09-07 144714" src="https://github.com/user-attachments/assets/3b872b84-86f5-4e20-8552-2d99faead793" />


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

* **Monte Carlo Sampling** — Introduce randomized multi-sample ray tracing to reduce noise and improve image quality.
* **Progressive Rendering** — Accumulate samples across frames to progressively converge toward a cleaner image.
* **BVH Acceleration** — Implement a Bounding Volume Hierarchy to significantly reduce ray-scene intersection costs.
* **Triangle Meshes** — Extend the renderer beyond spheres to support arbitrary triangle-based geometry and 3D models.
* **Importance Sampling** — Improve sampling efficiency by preferentially sampling directions that contribute more to the final image.
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
