#ifndef __OPENCL_VERSION__
#include "../lib/OpenCL/OpenCLKernel.hpp"  // Hack to make syntax highlighting work
#endif

// OpenCL kernel for Marching Cubes

// Normals computation kernel: Computes the gradient of the volume for use in shading
__kernel void compute_normals(__global const float* volume, __global float* normals, int width, int height, int depth) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    int z = get_global_id(2);

    // Skip border voxels (avoid out-of-bounds access)
    if (x == 0 || y == 0 || z == 0 || x == width - 1 || y == height - 1 || z == depth - 1) {
        return;
    }

    // Compute the central differences for gradient approximation
    float dx = (volume[(x + 1) + width * (y + height * z)] - volume[(x - 1) + width * (y + height * z)]) * 0.5f;
    float dy = (volume[x + width * ((y + 1) + height * z)] - volume[x + width * ((y - 1) + height * z)]) * 0.5f;
    float dz = (volume[x + width * (y + height * (z + 1))] - volume[x + width * (y + height * (z - 1))]) * 0.5f;

    // Store the computed normals in the global buffer
    int idx = 3 * (x + width * (y + height * z));
    normals[idx + 0] = -dx;
    normals[idx + 1] = -dy;
    normals[idx + 2] = -dz;
}

// Helper function to interpolate the vertex position along an edge
float3 interpolate_vertex(float isoValue, float3 p0, float3 p1, float val0, float val1) {
    float t = (isoValue - val0) / (val1 - val0);
    return p0 + t * (p1 - p0);
}

// Marching Cubes kernel: Processes each voxel and generates triangles based on isosurface intersection
__kernel void marching_cubes(__global const float* volume,
                             __global float* vertices,
                             __global const ushort* edgeTable,
                             __global const int* triTable,
                             int width, int height, int depth, float isoValue) {

    // Get global thread indices corresponding to voxel coordinates
    int x = get_global_id(0);
    int y = get_global_id(1);
    int z = get_global_id(2);

    // Ensure we're not accessing out-of-bounds voxels
    if (x >= width - 1 || y >= height - 1 || z >= depth - 1) {
        return;
    }

    // Get voxel values for the 8 corners of the cube
    float cube[8];
    cube[0] = volume[(x) + width * ((y) + height * (z))];
    cube[1] = volume[(x + 1) + width * ((y) + height * (z))];
    cube[2] = volume[(x + 1) + width * ((y + 1) + height * (z))];
    cube[3] = volume[(x) + width * ((y + 1) + height * (z))];
    cube[4] = volume[(x) + width * ((y) + height * (z + 1))];
    cube[5] = volume[(x + 1) + width * ((y) + height * (z + 1))];
    cube[6] = volume[(x + 1) + width * ((y + 1) + height * (z + 1))];
    cube[7] = volume[(x) + width * ((y + 1) + height * (z + 1))];

    // Determine the index into the edge table which tells us which vertices are inside the surface
    int cubeIndex = 0;
    if (cube[0] < isoValue) cubeIndex |= 1;
    if (cube[1] < isoValue) cubeIndex |= 2;
    if (cube[2] < isoValue) cubeIndex |= 4;
    if (cube[3] < isoValue) cubeIndex |= 8;
    if (cube[4] < isoValue) cubeIndex |= 16;
    if (cube[5] < isoValue) cubeIndex |= 32;
    if (cube[6] < isoValue) cubeIndex |= 64;
    if (cube[7] < isoValue) cubeIndex |= 128;

    // Lookup the edges intersected by the isosurface using the cubeIndex
    int edgeFlags = edgeTable[cubeIndex];
    if (edgeFlags == 0) return;

    // Vertex positions for the 8 corners of the cube
    float3 p[8];
    p[0] = (float3)(x, y, z);
    p[1] = (float3)(x + 1, y, z);
    p[2] = (float3)(x + 1, y + 1, z);
    p[3] = (float3)(x, y + 1, z);
    p[4] = (float3)(x, y, z + 1);
    p[5] = (float3)(x + 1, y, z + 1);
    p[6] = (float3)(x + 1, y + 1, z + 1);
    p[7] = (float3)(x, y + 1, z + 1);

    // Interpolated vertex positions
    float3 vertList[12];

    // Interpolate vertices along the cube's edges
    if (edgeFlags & 1) vertList[0] = interpolate_vertex(isoValue, p[0], p[1], cube[0], cube[1]);
    if (edgeFlags & 2) vertList[1] = interpolate_vertex(isoValue, p[1], p[2], cube[1], cube[2]);
    if (edgeFlags & 4) vertList[2] = interpolate_vertex(isoValue, p[2], p[3], cube[2], cube[3]);
    if (edgeFlags & 8) vertList[3] = interpolate_vertex(isoValue, p[3], p[0], cube[3], cube[0]);
    if (edgeFlags & 16) vertList[4] = interpolate_vertex(isoValue, p[4], p[5], cube[4], cube[5]);
    if (edgeFlags & 32) vertList[5] = interpolate_vertex(isoValue, p[5], p[6], cube[5], cube[6]);
    if (edgeFlags & 64) vertList[6] = interpolate_vertex(isoValue, p[6], p[7], cube[6], cube[7]);
    if (edgeFlags & 128) vertList[7] = interpolate_vertex(isoValue, p[7], p[4], cube[7], cube[4]);
    if (edgeFlags & 256) vertList[8] = interpolate_vertex(isoValue, p[0], p[4], cube[0], cube[4]);
    if (edgeFlags & 512) vertList[9] = interpolate_vertex(isoValue, p[1], p[5], cube[1], cube[5]);
    if (edgeFlags & 1024) vertList[10] = interpolate_vertex(isoValue, p[2], p[6], cube[2], cube[6]);
    if (edgeFlags & 2048) vertList[11] = interpolate_vertex(isoValue, p[3], p[7], cube[3], cube[7]);

    // Output triangles formed by intersected edges
    int index = get_global_id(0) * 15 * 3; // Each voxel can contribute up to 15 vertices

    for (int i = 0; triTable[cubeIndex * 16 + i] != -1; i += 3) { // Flattened access to triTable
        int vertexA = triTable[cubeIndex * 16 + i];
        int vertexB = triTable[cubeIndex * 16 + i + 1];
        int vertexC = triTable[cubeIndex * 16 + i + 2];

        // Store the triangle's vertices in the global buffer
        vertices[index++] = vertList[vertexA].x;
        vertices[index++] = vertList[vertexA].y;
        vertices[index++] = vertList[vertexA].z;

        vertices[index++] = vertList[vertexB].x;
        vertices[index++] = vertList[vertexB].y;
        vertices[index++] = vertList[vertexB].z;

        vertices[index++] = vertList[vertexC].x;
        vertices[index++] = vertList[vertexC].y;
        vertices[index++] = vertList[vertexC].z;
    }
}
