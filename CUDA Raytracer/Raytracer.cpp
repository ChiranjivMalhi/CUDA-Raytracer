#include <cuda_runtime.h>

#include "hittable.h"
#include "hittable_list.h"
#include "sphere.h"
#include "constants.h"
#include "camera.h"
#include "material_types.h"
#include "cuda_kernels.cuh"
#include "memory.cuh"
#include "obj_mesh.h"

#include <glfw3.h>
#include <vector>
#include <string>
#include <iostream>
#include <ctime>
#include <exception>

int main(int argc, char** argv) {
    std::string model_path;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--model" && i + 1 < argc) {
            model_path = argv[++i];
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: CUDA Raytracer.exe [--model path/to/model.obj]\n"
                         "Without --model, the procedural sphere scene is rendered.\n";
            return 0;
        } else {
            std::cerr << "Unknown or incomplete option: " << arg << "\n"
                      << "Usage: CUDA Raytracer.exe [--model path/to/model.obj]\n";
            return 1;
        }
    }
    
    std::vector<GPUSphere> gpu_spheres;
    std::vector<material> gpu_materials;
    ObjMesh mesh;

    gpu_materials.push_back(material(color(0.5, 0.5, 0.5), material_type::lambertian));
    auto ground_material_id = 0;
    gpu_spheres.push_back(GPUSphere{point3(0, -1000, 0), 1000.0, ground_material_id});

    camera cam;
    if (!model_path.empty()) {
        try {
            const int diffuse_id = static_cast<int>(gpu_materials.size());
            gpu_materials.push_back(material(color(0.78, 0.24, 0.10), material_type::lambertian));

            const int metal_id = static_cast<int>(gpu_materials.size());
            material metal(color(0.72, 0.76, 0.82), material_type::metal);
            metal.fuzz = 0.03;
            gpu_materials.push_back(metal);

            const int glass_id = static_cast<int>(gpu_materials.size());
            gpu_materials.push_back(material(1.5));

            mesh.load(model_path, { diffuse_id, metal_id, glass_id });
        } catch (const std::exception& error) {
            std::cerr << "Could not load model: " << error.what() << "\n";
            return 1;
        }
        cam.lookfrom = point3(0, 4.0, 15.0);
        cam.lookat = point3(0, 1.2, 0);
        cam.vup = vec3(0, 1, 0);
        cam.vfov = 36;
        cam.focus_dist = 15.3;
        cam.defocus_angle = 0.0;
        std::cout << "Loaded OBJ " << model_path << " as three material variants ("
                  << mesh.triangles.size() << " triangles total, " << mesh.nodes.size()
                  << " BVH nodes).\n";
    } else {
        int mat_id = 1;
        for (int a = -11; a < 11; a++) {
            for (int b = -11; b < 11; b++) {
                auto choose_mat = random_double();
                point3 center(a + 0.9 * random_double(), 0.2, b + 0.9 * random_double());
                if ((center - point3(4, 0.2, 0)).length() > 0.9) {
                    int sphere_mat_id;
                    if (choose_mat < 0.8) {
                        gpu_materials.push_back(material(color::random() * color::random(), material_type::lambertian));
                        sphere_mat_id = mat_id++;
                    } else if (choose_mat < 0.95) {
                        material m(color::random(0.5, 1), material_type::metal);
                        m.fuzz = random_double(0, 0.5);
                        gpu_materials.push_back(m);
                        sphere_mat_id = mat_id++;
                    } else {
                        gpu_materials.push_back(material(1.5));
                        sphere_mat_id = mat_id++;
                    }
                    gpu_spheres.push_back(GPUSphere{center, 0.2, sphere_mat_id});
                }
            }
        }
        gpu_materials.push_back(material(1.5));
        const int glass_mat_id = mat_id++;
        gpu_spheres.push_back(GPUSphere{point3(0, 1, 0), 1.0, glass_mat_id});
        gpu_materials.push_back(material(color(0.4, 0.2, 0.1), material_type::lambertian));
        const int brown_mat_id = mat_id++;
        gpu_spheres.push_back(GPUSphere{point3(-4, 1, 0), 1.0, brown_mat_id});
        gpu_materials.push_back(material(color(0.7, 0.6, 0.5), material_type::metal));
        const int metal_mat_id = mat_id++;
        gpu_materials[metal_mat_id].fuzz = 0.0;
        gpu_spheres.push_back(GPUSphere{point3(4, 1, 0), 1.0, metal_mat_id});
        std::cout << "Loaded " << gpu_spheres.size() << " spheres and " << gpu_materials.size() << " materials.\n";
    }
    
    cam.aspect_ratio = 16.0 / 9.0;
    cam.image_width = 540;
    cam.samples_per_pixel = 2;
    cam.max_depth = 5;
    cam.vfov = model_path.empty() ? 20 : 36;
    if (model_path.empty()) {
        cam.lookfrom = point3(13, 2, 3);
        cam.lookat = point3(0, 0, 0);
        cam.vup = vec3(0, 1, 0);
        cam.defocus_angle = 0.6;
        cam.focus_dist = 10.0;
    }
   
    cam.prepare_progressive_render();
    
    if (!glfwInit()) {
        std::cerr << "Could not initialize GLFW.\n";
        return 1;
    }
    
    GLFWwindow* window = glfwCreateWindow(960, 540, "CUDA Ray Tracer", nullptr, nullptr);
    if (window == nullptr) {
        std::cerr << "Could not create GLFW window.\n";
        glfwTerminate();
        return 1;
    }
    
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);
    
    GLuint image_texture = 0;
    glGenTextures(1, &image_texture);
    glBindTexture(GL_TEXTURE_2D, image_texture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, cam.image_width, cam.rendered_image_height(),
                 0, GL_RGB, GL_UNSIGNED_BYTE, nullptr);
    
    GPUScene gpu_scene;
    gpu_scene.allocate(cam.image_width, cam.rendered_image_height(),
                       static_cast<int>(gpu_spheres.size()), static_cast<int>(gpu_materials.size()),
                       static_cast<int>(mesh.triangles.size()), static_cast<int>(mesh.nodes.size()));
    gpu_scene.transfer_scene_data(gpu_spheres, gpu_materials, mesh.triangles, mesh.nodes, mesh.triangle_indices);
    
    CameraGPU gpu_cam;
    gpu_cam.lookfrom = cam.lookfrom;
    gpu_cam.image_width = cam.image_width;
    gpu_cam.image_height = cam.rendered_image_height();
    gpu_cam.max_depth = cam.max_depth;
    gpu_cam.pixel00_loc = cam.pixel00_loc;
    gpu_cam.pixel_delta_u = cam.pixel_delta_u;
    gpu_cam.pixel_delta_v = cam.pixel_delta_v;
    
    cudaMemcpy(gpu_scene.d_camera, &gpu_cam, sizeof(CameraGPU), cudaMemcpyHostToDevice);
    
    dim3 threads(256);
    dim3 blocks((cam.image_width * cam.rendered_image_height() + 255) / 256);
    init_curand_kernel<<<blocks, threads>>>(gpu_scene.d_rand_states, 
                                             (unsigned long long)time(nullptr), 
                                             cam.image_width * cam.rendered_image_height());
    
    int accumulated_samples = 0;
    std::vector<uint8_t> image(cam.image_width * cam.rendered_image_height() * 3);
    
    glfwSetWindowTitle(window, "CUDA Ray Tracer - Rendering...");
    
    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();

        ray_tracing_kernel<<<blocks, threads>>>(
            gpu_scene.d_accumulation,
            gpu_scene.d_spheres,
            gpu_scene.num_spheres,
            gpu_scene.d_triangles,
            gpu_scene.d_bvh_nodes,
            gpu_scene.num_bvh_nodes,
            gpu_scene.d_triangle_indices,
            gpu_scene.d_materials,
            gpu_scene.d_camera,
            gpu_scene.d_rand_states,
            accumulated_samples);
        
        accumulated_samples++;

        dim3 blocks2d((cam.image_width + 15) / 16, (cam.rendered_image_height() + 15) / 16);
        dim3 threads2d(16, 16);
        tone_mapping_kernel<<<blocks2d, threads2d>>>(
            gpu_scene.d_output_image,
            gpu_scene.d_accumulation,
            accumulated_samples,
            cam.image_width,
            cam.rendered_image_height());

        gpu_scene.get_output_image(image);
        glBindTexture(GL_TEXTURE_2D, image_texture);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, cam.image_width, cam.rendered_image_height(),
                        GL_RGB, GL_UNSIGNED_BYTE, image.data());

        int framebuffer_width = 0, framebuffer_height = 0;
        glfwGetFramebufferSize(window, &framebuffer_width, &framebuffer_height);
        glViewport(0, 0, framebuffer_width, framebuffer_height);
        glClearColor(0.08f, 0.12f, 0.20f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        
        glEnable(GL_TEXTURE_2D);
        glBindTexture(GL_TEXTURE_2D, image_texture);
        glBegin(GL_QUADS);
        glTexCoord2f(0.0f, 1.0f); glVertex2f(-1.0f, -1.0f);
        glTexCoord2f(1.0f, 1.0f); glVertex2f(1.0f, -1.0f);
        glTexCoord2f(1.0f, 0.0f); glVertex2f(1.0f, 1.0f);
        glTexCoord2f(0.0f, 0.0f); glVertex2f(-1.0f, 1.0f);
        glEnd();
        glDisable(GL_TEXTURE_2D);
        
        glfwSwapBuffers(window);

        std::string title = "CUDA Ray Tracer - " + std::to_string(accumulated_samples) + " samples";
        glfwSetWindowTitle(window, title.c_str());
    }

    glDeleteTextures(1, &image_texture);
    glfwDestroyWindow(window);
    glfwTerminate();
    
    return 0;
}
