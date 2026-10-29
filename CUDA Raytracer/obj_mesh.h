#ifndef OBJ_MESH_H
#define OBJ_MESH_H

#include "cuda_kernels.cuh"

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

// Small OBJ reader and CPU median-split BVH builder. It intentionally supports
// positions and polygon faces only; face normals are computed by the GPU.
class ObjMesh {
public:
    std::vector<GPUTriangle> triangles;
    std::vector<GPUBvhNode> nodes;
    std::vector<int> triangle_indices;

    void load(const std::string& path, const std::array<int, 3>& material_ids) {
        std::ifstream input(path);
        if (!input) {
            throw std::runtime_error("Could not open OBJ file: " + path);
        }

        std::vector<point3> vertices;
        std::string line;
        int line_number = 0;
        while (std::getline(input, line)) {
            ++line_number;
            std::istringstream row(line);
            std::string tag;
            row >> tag;
            if (tag.empty() || tag[0] == '#') continue;

            if (tag == "v") {
                double x, y, z;
                if (!(row >> x >> y >> z)) {
                    throw std::runtime_error("Malformed vertex at OBJ line " + std::to_string(line_number));
                }
                vertices.emplace_back(x, y, z);
            } else if (tag == "f") {
                std::vector<int> face;
                std::string token;
                while (row >> token) {
                    if (!token.empty() && token[0] == '#') break;
                    const auto slash = token.find('/');
                    const std::string vertex_part = token.substr(0, slash);
                    if (vertex_part.empty()) {
                        throw std::runtime_error("Missing vertex index at OBJ line " + std::to_string(line_number));
                    }

                    const int obj_index = std::stoi(vertex_part);
                    const int vertex_index = obj_index > 0
                        ? obj_index - 1
                        : static_cast<int>(vertices.size()) + obj_index;
                    if (vertex_index < 0 || vertex_index >= static_cast<int>(vertices.size())) {
                        throw std::runtime_error("Vertex index out of range at OBJ line " + std::to_string(line_number));
                    }
                    face.push_back(vertex_index);
                }

                if (face.size() < 3) {
                    throw std::runtime_error("Face has fewer than three vertices at OBJ line " + std::to_string(line_number));
                }

                // OBJ permits polygons. Fan triangulation handles convex faces
                // and the common triangular/quadrilateral model cases.
                for (size_t i = 1; i + 1 < face.size(); ++i) {
                    triangles.push_back(GPUTriangle{
                        vertices[face[0]], vertices[face[i]], vertices[face[i + 1]], material_ids[0]
                    });
                }
            }
        }

        if (triangles.empty()) {
            throw std::runtime_error("OBJ contains no faces: " + path);
        }

        normalize_to_ground();
        create_three_instances(material_ids);
        build_bvh();
    }

private:
    static point3 component_min(const point3& a, const point3& b) {
        return point3(std::fmin(a.x(), b.x()), std::fmin(a.y(), b.y()), std::fmin(a.z(), b.z()));
    }

    static point3 component_max(const point3& a, const point3& b) {
        return point3(std::fmax(a.x(), b.x()), std::fmax(a.y(), b.y()), std::fmax(a.z(), b.z()));
    }

    void normalize_to_ground() {
        point3 lo = triangles[0].v0;
        point3 hi = lo;
        for (const auto& tri : triangles) {
            lo = component_min(lo, component_min(tri.v0, component_min(tri.v1, tri.v2)));
            hi = component_max(hi, component_max(tri.v0, component_max(tri.v1, tri.v2)));
        }

        const point3 extent = hi - lo;
        const double largest_extent = std::fmax(extent.x(), std::fmax(extent.y(), extent.z()));
        if (!(largest_extent > 1e-12) || !std::isfinite(largest_extent)) {
            throw std::runtime_error("OBJ has invalid or zero-sized bounds");
        }

        // Center horizontally, place the lowest point on y=0, and fit the
        // longest dimension into a three-world-unit range.
        const double scale = 3.0 / largest_extent;
        const point3 center((lo.x() + hi.x()) * 0.5, lo.y(), (lo.z() + hi.z()) * 0.5);
        for (auto& tri : triangles) {
            tri.v0 = scale * (tri.v0 - center);
            tri.v1 = scale * (tri.v1 - center);
            tri.v2 = scale * (tri.v2 - center);
        }
    }

    void create_three_instances(const std::array<int, 3>& material_ids) {
        const std::vector<GPUTriangle> normalized_model = triangles;
        triangles.clear();
        triangles.reserve(normalized_model.size() * 3);

        // Leave a visible gap between copies of the normalized model.
        const double x_offsets[3] = { -3.6, 0.0, 3.6 };
        for (int instance = 0; instance < 3; ++instance) {
            for (const auto& source : normalized_model) {
                GPUTriangle tri = source;
                const vec3 offset(x_offsets[instance], 0.0, 0.0);
                tri.v0 += offset;
                tri.v1 += offset;
                tri.v2 += offset;
                tri.material_id = material_ids[instance];
                triangles.push_back(tri);
            }
        }
    }

    point3 triangle_min(int index) const {
        const auto& tri = triangles[index];
        return component_min(tri.v0, component_min(tri.v1, tri.v2));
    }

    point3 triangle_max(int index) const {
        const auto& tri = triangles[index];
        return component_max(tri.v0, component_max(tri.v1, tri.v2));
    }

    point3 triangle_centroid(int index) const {
        const auto& tri = triangles[index];
        return (tri.v0 + tri.v1 + tri.v2) / 3.0;
    }

    int build_node(size_t begin, size_t end) {
        const int node_index = static_cast<int>(nodes.size());
        nodes.emplace_back();

        point3 bounds_min = triangle_min(triangle_indices[begin]);
        point3 bounds_max = triangle_max(triangle_indices[begin]);
        point3 centroid_min = triangle_centroid(triangle_indices[begin]);
        point3 centroid_max = centroid_min;
        for (size_t i = begin + 1; i < end; ++i) {
            const int tri_index = triangle_indices[i];
            bounds_min = component_min(bounds_min, triangle_min(tri_index));
            bounds_max = component_max(bounds_max, triangle_max(tri_index));
            const point3 c = triangle_centroid(tri_index);
            centroid_min = component_min(centroid_min, c);
            centroid_max = component_max(centroid_max, c);
        }

        GPUBvhNode node{};
        node.bounds_min = bounds_min;
        node.bounds_max = bounds_max;
        node.left = -1;
        node.right = -1;
        node.first = static_cast<int>(begin);
        node.count = static_cast<int>(end - begin);

        constexpr size_t leaf_size = 4;
        if (end - begin > leaf_size) {
            const point3 span = centroid_max - centroid_min;
            int axis = span.y() > span.x() ? 1 : 0;
            if (span.z() > span[axis]) axis = 2;

            const size_t middle = begin + (end - begin) / 2;
            std::nth_element(triangle_indices.begin() + begin, triangle_indices.begin() + middle,
                triangle_indices.begin() + end, [this, axis](int a, int b) {
                    return triangle_centroid(a)[axis] < triangle_centroid(b)[axis];
                });

            const int left = build_node(begin, middle);
            const int right = build_node(middle, end);
            node.left = left;
            node.right = right;
            node.count = 0;
        }

        nodes[node_index] = node;
        return node_index;
    }

    void build_bvh() {
        triangle_indices.resize(triangles.size());
        for (size_t i = 0; i < triangle_indices.size(); ++i) {
            triangle_indices[i] = static_cast<int>(i);
        }
        nodes.clear();
        nodes.reserve(triangles.size() * 2);
        build_node(0, triangle_indices.size());
    }
};

#endif
