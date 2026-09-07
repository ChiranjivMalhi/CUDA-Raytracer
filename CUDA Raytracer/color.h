#ifndef COLOR_H
#define COLOR_H

#include "vec3.h"
#include "interval.h"

#include <array>
#include <cstdint>
#include <iostream>

using color = vec3;

inline double linear_to_gamma(double linear_component) {
    if (linear_component > 0)
        return std::sqrt(linear_component);
    return 0;
}

inline std::array<std::uint8_t, 3> color_to_rgb(const color& pixel_color) {
    auto r = pixel_color.x();
    auto g = pixel_color.y();
    auto b = pixel_color.z();

    r = linear_to_gamma(r);
    g = linear_to_gamma(g);
    b = linear_to_gamma(b);

    static const interval intensity(0.000, 0.999);
    auto rbyte = static_cast<std::uint8_t>(256 * intensity.clamp(r));
    auto gbyte = static_cast<std::uint8_t>(256 * intensity.clamp(g));
    auto bbyte = static_cast<std::uint8_t>(256 * intensity.clamp(b));

    return {rbyte, gbyte, bbyte};
}

inline void write_color(std::ostream& out, const color& pixel_color) {
    const auto [rbyte, gbyte, bbyte] = color_to_rgb(pixel_color);
    out << rbyte << ' ' << gbyte << ' ' << bbyte << '\n';
}

#endif
