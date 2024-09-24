// Includes
#include <GL/glew.h>
#include <GL/freeglut.h>
// Note: GL/glew.h must be included before other OpenGL files
#include <GL/glx.h>
#include <OpenCL/cl-patched.hpp>
#include <hdf5.h>
#include <fstream>
#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include "Lut.hpp" // Lookup table for Marching Cubes
#include <algorithm> // Add this to resolve std::max
#include <array>  // <array> included here

// Vec3 structure to replace glm::vec3
struct Vec3 {
    float x, y, z;

    Vec3() : x(0), y(0), z(0) {}
    Vec3(float x_, float y_, float z_) : x(x_), y(y_), z(z_) {}

    // Add two vectors
    Vec3 operator+(const Vec3& other) const {
        return Vec3(x + other.x, y + other.y, z + other.z);
    }

    // Subtract two vectors
    Vec3 operator-(const Vec3& other) const {
        return Vec3(x - other.x, y - other.y, z - other.z);
    }

    // Scalar multiplication
    Vec3 operator*(float scalar) const {
        return Vec3(x * scalar, y * scalar, z * scalar);
    }
};

// Global Variables
cl::Context context;
cl::CommandQueue queue;
GLuint va, vbo, shaderProgram = 0;
std::vector<cl_float> h_volume_in, h_normals, h_va_out_cpu, h_va_out_gpu;
int volumeRes[3];
float volumeDim[3];
unsigned int dataCount, cellCount, outputCount;
cl_float isoVal;
bool runCpu = true, runGpu = true, displayGpu = true;
float rotX, rotY, rotZ = 0;

// Function Declarations
//void openGLsetArgsup();
void openGLSetup();
void displayGL();
void keyboardGL(unsigned char key, int x, int y);
void setIdle();
void idleGL();
double runCPU();
std::array<double, 2> runGPU();
void checkConsistency();
void createOpenCLContext();
float* interpolateEdge(float isoVal, float val0, float val1, int idx0, int idx1, int x, int y, int z);

// Helper Function for Time Measurement
double currentTimeInMicroseconds() {
    auto now = std::chrono::high_resolution_clock::now();
    auto micros = std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch()).count();
    return static_cast<double>(micros);
}

// Main Function
int main(int argc, char** argv) {
    // Initialize volume data from .hdf5 file
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <volume file> <isoValue>" << std::endl;
        return -1;
    }

    // Set isoValue and load volume
    isoVal = std::stof(argv[2]);
    hid_t file_id = H5Fopen(argv[1], H5F_ACC_RDONLY, H5P_DEFAULT);
    hid_t dataset = H5Dopen(file_id, "Volume", H5P_DEFAULT);
    hid_t dataspace = H5Dget_space(dataset);

    hsize_t dims[3];
    H5Sget_simple_extent_dims(dataspace, dims, NULL);
    volumeRes[0] = dims[0];
    volumeRes[1] = dims[1];
    volumeRes[2] = dims[2];

    dataCount = volumeRes[0] * volumeRes[1] * volumeRes[2];
    h_volume_in.resize(dataCount);
    H5Dread(dataset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, h_volume_in.data());
    H5Dclose(dataset);
    H5Fclose(file_id);

    float maxDim = std::max({volumeRes[0], volumeRes[1], volumeRes[2]});
    volumeDim[0] = 1.0;
    volumeDim[1] = 1.0;
    volumeDim[2] = 1.0;

    // Prepare volume and surface data
    cellCount = (volumeRes[0] - 1) * (volumeRes[1] - 1) * (volumeRes[2] - 1);
    outputCount = 6 * 15 * cellCount;
    h_va_out_cpu.resize(outputCount, 0);
    h_va_out_gpu.resize(outputCount, 0);
    h_normals.resize(3 * dataCount, 0);

    // Normalize data
    float minVal = *std::min_element(h_volume_in.begin(), h_volume_in.end());
    float maxVal = *std::max_element(h_volume_in.begin(), h_volume_in.end());
    for (auto& val : h_volume_in) {
        val = (val - minVal) / (maxVal - minVal);
    }

    // Run CPU and GPU versions
    double cpuTime = runCpu ? runCPU() : 0;
    std::array<double, 2> gpuTimes = runGpu ? runGPU() : std::array<double, 2>{0, 0};

    // Output performance data
    std::cout << "Data points: " << dataCount << " (" << volumeRes[0] << "x" << volumeRes[1] << "x" << volumeRes[2] << ")\n";
    if (runCpu) std::cout << "Calc time CPU: " << cpuTime << " µs\n";
    if (runGpu) {
        std::cout << "Calc time GPU: " << gpuTimes[0] << " µs\n";
        std::cout << "Mem transfer time GPU: " << gpuTimes[1] << " µs\n";
        if (runCpu) {
            std::cout << "Calc-only speedup: " << cpuTime / gpuTimes[0] << " times\n";
            std::cout << "Overall speedup: " << cpuTime / (gpuTimes[0] + gpuTimes[1]) << " times\n";
        }
    }

    // OpenGL Visualization
    if (runGpu && displayGpu) {
        glutInit(&argc, argv);
        openGLSetup();
        glutMainLoop();
    }

    return 0;
}

// CPU Implementation (Marching Cubes)
double runCPU() {
    double start = currentTimeInMicroseconds();

    int vertexOffset = 0;
    for (int z = 0; z < volumeRes[2] - 1; z++) {
        for (int y = 0; y < volumeRes[1] - 1; y++) {
            for (int x = 0; x < volumeRes[0] - 1; x++) {
                int cubeIndex = 0;
                float cubeValues[8];

                // Get the 8 values in the cube from the volume
                for (int i = 0; i < 8; i++) {
                    int xi = x + (i & 1);
                    int yi = y + ((i >> 1) & 1);
                    int zi = z + ((i >> 2) & 1);
                    cubeValues[i] = h_volume_in[xi + volumeRes[0] * (yi + volumeRes[1] * zi)];
                    if (cubeValues[i] < isoVal) {
                        cubeIndex |= 1 << i;
                    }
                }

                // Find which edges are intersected by the isosurface
                int edgeFlags = edgeTable[cubeIndex];
                if (edgeFlags == 0) continue;

                // Interpolate the vertex positions where the surface intersects the cube edges
                //Vec3 vertexList[12];
                float vertexList[12][3];  // 12 edges, and each edge has a 3D point (x, y, z)
                for (int i = 0; i < 12; i++) {
                    if (edgeFlags & (1 << i)) {
                        int idx0 = edgeToVertex[i][0];
                        int idx1 = edgeToVertex[i][1];
                        float val0 = cubeValues[idx0];
                        float val1 = cubeValues[idx1];
                        //vertexList[i] = interpolateEdge(isoVal, val0, val1, idx0, idx1, x, y, z);
                                // Interpolate the position where the surface intersects the edge
                        float* vertexPos = interpolateEdge(isoVal, val0, val1, idx0, idx1, x, y, z);

                        // Assign the 3D position to vertexList
                        vertexList[i][0] = vertexPos[0];  // x coordinate
                        vertexList[i][1] = vertexPos[1];  // y coordinate
                        vertexList[i][2] = vertexPos[2];  // z coordinate
                    }
                }

                // Create triangles
                for (int i = 0; triTable[cubeIndex][i] != -1; i += 3) {
                      h_va_out_cpu[vertexOffset++] = vertexList[triTable[cubeIndex][i]][0];   // x component
                      h_va_out_cpu[vertexOffset++] = vertexList[triTable[cubeIndex][i + 1]][1]; // y component
                      h_va_out_cpu[vertexOffset++] = vertexList[triTable[cubeIndex][i + 2]][2]; // z component
                }

            }
        }
    }

    return currentTimeInMicroseconds() - start;
}






// GPU Implementation (Marching Cubes with OpenCL)
std::array<double, 2> runGPU() {
    // Create OpenCL context
    createOpenCLContext();

    // Flatten the 2D `triTable` into a 1D array for OpenCL
    std::vector<int> flatTriTable;
    for (const auto& row : triTable) {
      flatTriTable.insert(flatTriTable.end(), row.begin(), row.end());
    }

    // Create buffers for the volume and output
    cl::Buffer volumeBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, dataCount * sizeof(cl_float), h_volume_in.data());
    cl::Buffer normalBuffer(context, CL_MEM_READ_WRITE, dataCount * 3 * sizeof(cl_float));
    cl::Buffer vertexBuffer(context, CL_MEM_WRITE_ONLY, outputCount * sizeof(cl_float));
    cl::Buffer triTableBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, flatTriTable.size() * sizeof(int), flatTriTable.data());
    cl::Buffer edgeTableBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, edgeTable.size() * sizeof(cl_ushort),  edgeTable.data());
    //cl::Buffer edgeTableBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(edgeTable), edgeTable);
    //cl::Buffer triTableBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(triTable), triTable);

    // Load OpenCL program and create kernels (normals and marching cubes)
    std::ifstream sourceFile("/zhome/annavanr/OpenCLExercise4_VolumeRendering/src/OpenCLExercise4_VolumeRendering.cl");
    if (!sourceFile.is_open()) {
    std::cerr << "Failed to load OpenCL source file" << std::endl;
    }
    std::string sourceCode(std::istreambuf_iterator<char>(sourceFile), (std::istreambuf_iterator<char>()));
    cl::Program::Sources source(1, std::make_pair(sourceCode.c_str(), sourceCode.length() + 1));
    cl::Program program(context, source);
    program.build();



    // Create kernels
    cl::Kernel normKernel(program, "compute_normals");
    cl::Kernel mcKernel(program, "marching_cubes");

    cl_int err;
    err = mcKernel.setArg(0, volumeBuffer);      // Volume buffer (input)
    if (err != CL_SUCCESS) std::cerr << "Error setting arg 0: " << err << std::endl;
    err = mcKernel.setArg(1, vertexBuffer);      // Vertex buffer (output)
    if (err != CL_SUCCESS) std::cerr << "Error setting arg 1: " << err << std::endl;
    err = mcKernel.setArg(2, edgeTableBuffer);   // Edge table buffer (input)
    if (err != CL_SUCCESS) std::cerr << "Error setting arg 2: " << err << std::endl;
    err = mcKernel.setArg(3, triTableBuffer);    // Tri table buffer (input)
    if (err != CL_SUCCESS) std::cerr << "Error setting arg 3: " << err << std::endl;
    err=mcKernel.setArg(4, volumeRes[0]);             // Volume width
    if (err != CL_SUCCESS) std::cerr << "Error setting arg 3: " << err << std::endl;
    err=mcKernel.setArg(5, volumeRes[1]);            // Volume height
    if (err != CL_SUCCESS) std::cerr << "Error setting arg 3: " << err << std::endl;
    err=mcKernel.setArg(6, volumeRes[2]);             // Volume depth
    if (err != CL_SUCCESS) std::cerr << "Error setting arg 3: " << err << std::endl;
    err=mcKernel.setArg(7, isoVal);
    if (err != CL_SUCCESS) std::cerr << "Error setting arg 3: " << err << std::endl;// Isosurface value




    err = normKernel.setArg(0, volumeBuffer);      // Volume buffer (input)
    if (err != CL_SUCCESS) std::cerr << "Error setting arg 0: " << err << std::endl;
    err = normKernel.setArg(1, normalBuffer);      // Vertex buffer (output)
    if (err != CL_SUCCESS) std::cerr << "Error setting arg 1: " << err << std::endl;
    err = normKernel.setArg(2, volumeRes[0]);   // Edge table buffer (input)
    if (err != CL_SUCCESS) std::cerr << "Error setting arg 2: " << err << std::endl;
    err = normKernel.setArg(3, volumeRes[1]);    // Tri table buffer (input)
    if (err != CL_SUCCESS) std::cerr << "Error setting arg 3: " << err << std::endl;
    err=normKernel.setArg(4, volumeRes[2]);             // Volume width
    if (err != CL_SUCCESS) std::cerr << "Error setting arg 3: " << err << std::endl;

    // Normals kernel execution
    double start = currentTimeInMicroseconds();
    queue.enqueueNDRangeKernel(normKernel, cl::NullRange, cl::NDRange(volumeRes[0], volumeRes[1], volumeRes[2]));
    queue.finish();

    // Marching Cubes kernel execution

    queue.enqueueNDRangeKernel(mcKernel, cl::NullRange, cl::NDRange(volumeRes[0] - 1, volumeRes[1] - 1, volumeRes[2] - 1));
    queue.finish();

    double calcTime = currentTimeInMicroseconds() - start;

    // Read back the vertices from GPU
    double transferStart = currentTimeInMicroseconds();
    queue.enqueueReadBuffer(vertexBuffer, CL_TRUE, 0, outputCount * sizeof(cl_float), h_va_out_gpu.data());
    double memTransferTime = currentTimeInMicroseconds() - transferStart;

    return {calcTime, memTransferTime};
}

void createOpenCLContext() {
    // Platform and device setup
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    context = cl::Context(CL_DEVICE_TYPE_GPU);
    std::vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();
    queue = cl::CommandQueue(context, devices[0]);
}

// Helper function to interpolate between cube edges
float* interpolateEdge(float isoVal, float val0, float val1, int idx0, int idx1, int x, int y, int z) {
    // Allocate memory for the interpolated position (as an array of 3 floats)
    static float result[3];

    // Calculate interpolation factor
    float t = (isoVal - val0) / (val1 - val0);

    // Interpolate between two points along the edge
    result[0] = (x + vertexOffset[idx0][0]) + t * (vertexOffset[idx1][0] - vertexOffset[idx0][0]);
    result[1] = (y + vertexOffset[idx0][1]) + t * (vertexOffset[idx1][1] - vertexOffset[idx0][1]);
    result[2] = (z + vertexOffset[idx0][2]) + t * (vertexOffset[idx1][2] - vertexOffset[idx0][2]);

    return result;
}


// Rendering setup
void openGLSetup() {
    glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH);
    glutCreateWindow("Marching Cubes on GPU");
    glewInit();

    glutDisplayFunc(displayGL);
    glutKeyboardFunc(keyboardGL);
    setIdle();
    glEnable(GL_DEPTH_TEST);
}

// Rendering function
void displayGL() {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // Render the vertices from CPU/GPU
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, outputCount * sizeof(cl_float), h_va_out_gpu.data(), GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(0);

    glDrawArrays(GL_TRIANGLES, 0, outputCount / 3);

    glutSwapBuffers();
}

void setIdle() { glutIdleFunc(idleGL); }

void idleGL() { displayGL(); }

void keyboardGL(unsigned char key, int x, int y) {
    if (key == 27) glutLeaveMainLoop(); // Escape key to exit
}
