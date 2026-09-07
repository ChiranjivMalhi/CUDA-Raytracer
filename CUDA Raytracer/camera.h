#ifndef CAMERA_H
#define CAMERA_H

#include "hittable.h"
#include "constants.h"
#include "vec3.h"

#include <cstdint>
#include <vector>
#include <cmath>

class camera {
public:
    double aspect_ratio = 1.0;
    int    image_width = 100;
    int    samples_per_pixel = 10;
    int    max_depth = 10;

    double vfov = 90;
    point3 lookfrom = point3(0, 0, 0);
    point3 lookat = point3(0, 0, -1);
    vec3   vup = vec3(0, 1, 0);

    double defocus_angle = 0;
    double focus_dist = 10;

    int    image_height = 1;
    point3 center;
    point3 pixel00_loc;
    vec3   pixel_delta_u;
    vec3   pixel_delta_v;

    void prepare_progressive_render() {
        initialize();
    }

    int rendered_image_height() const { return image_height; }

private:
    double pixel_samples_scale;
    vec3   u, v, w;
    vec3   defocus_disk_u;
    vec3   defocus_disk_v;

    void initialize() {
        image_height = static_cast<int>(image_width / aspect_ratio);
        if (image_height < 1) image_height = 1;

        pixel_samples_scale = 1.0 / samples_per_pixel;

        center = lookfrom;

        auto theta = degrees_to_radians(vfov);
        auto h = tan(theta / 2);
        auto viewport_height = 2.0 * h * focus_dist;
        auto viewport_width = viewport_height * (static_cast<double>(image_width) / image_height);

        w = unit_vector(lookfrom - lookat);
        u = unit_vector(cross(vup, w));
        v = cross(w, u);

        auto viewport_u = viewport_width * u;
        auto viewport_v = viewport_height * (-v);

        pixel_delta_u = viewport_u / image_width;
        pixel_delta_v = viewport_v / image_height;

        auto viewport_upper_left = center - (focus_dist * w) - viewport_u / 2 - viewport_v / 2;
        pixel00_loc = viewport_upper_left + 0.5 * (pixel_delta_u + pixel_delta_v);

        auto defocus_radius = focus_dist * tan(degrees_to_radians(defocus_angle / 2));
        defocus_disk_u = u * defocus_radius;
        defocus_disk_v = v * defocus_radius;
    }
};

#endif
