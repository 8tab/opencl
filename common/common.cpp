#include <iostream>
#include <vector>
#include <string>
#include <stdexcept>
#include <cstdlib>
#include <cstring>
#include <getopt.h>

#include "common.h"


const char* opencl_error_to_string(cl_int error)
{
        switch (error)
        {
                case CL_SUCCESS: return "CL_SUCCESS";
                case CL_DEVICE_NOT_FOUND: return "CL_DEVICE_NOT_FOUND";
                case CL_DEVICE_NOT_AVAILABLE: return "CL_DEVICE_NOT_AVAILABLE";
                case CL_COMPILER_NOT_AVAILABLE: return "CL_COMPILER_NOT_AVAILABLE";
                case CL_MEM_OBJECT_ALLOCATION_FAILURE: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
                case CL_OUT_OF_RESOURCES: return "CL_OUT_OF_RESOURCES";
                case CL_OUT_OF_HOST_MEMORY: return "CL_OUT_OF_HOST_MEMORY";
                case CL_PROFILING_INFO_NOT_AVAILABLE: return "CL_PROFILING_INFO_NOT_AVAILABLE";
                case CL_MEM_COPY_OVERLAP: return "CL_MEM_COPY_OVERLAP";
                case CL_IMAGE_FORMAT_MISMATCH: return "CL_IMAGE_FORMAT_MISMATCH";
                case CL_IMAGE_FORMAT_NOT_SUPPORTED: return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
                case CL_BUILD_PROGRAM_FAILURE: return "CL_BUILD_PROGRAM_FAILURE";
                case CL_MAP_FAILURE: return "CL_MAP_FAILURE";
                case CL_MISALIGNED_SUB_BUFFER_OFFSET: return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
                case CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST: return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
                case CL_COMPILE_PROGRAM_FAILURE: return "CL_COMPILE_PROGRAM_FAILURE";
                case CL_LINKER_NOT_AVAILABLE: return "CL_LINKER_NOT_AVAILABLE";
                case CL_LINK_PROGRAM_FAILURE: return "CL_LINK_PROGRAM_FAILURE";
                case CL_DEVICE_PARTITION_FAILED: return "CL_DEVICE_PARTITION_FAILED";
                case CL_KERNEL_ARG_INFO_NOT_AVAILABLE: return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";
                case CL_INVALID_VALUE: return "CL_INVALID_VALUE";
                case CL_INVALID_DEVICE_TYPE: return "CL_INVALID_DEVICE_TYPE";
                case CL_INVALID_PLATFORM: return "CL_INVALID_PLATFORM";
                case CL_INVALID_DEVICE: return "CL_INVALID_DEVICE";
                case CL_INVALID_CONTEXT: return "CL_INVALID_CONTEXT";
                case CL_INVALID_QUEUE_PROPERTIES: return "CL_INVALID_QUEUE_PROPERTIES";
                case CL_INVALID_COMMAND_QUEUE: return "CL_INVALID_COMMAND_QUEUE";
                case CL_INVALID_HOST_PTR: return "CL_INVALID_HOST_PTR";
                case CL_INVALID_MEM_OBJECT: return "CL_INVALID_MEM_OBJECT";
                case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR: return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
                case CL_INVALID_IMAGE_SIZE: return "CL_INVALID_IMAGE_SIZE";
                case CL_INVALID_SAMPLER: return "CL_INVALID_SAMPLER";
                case CL_INVALID_BINARY: return "CL_INVALID_BINARY";
                case CL_INVALID_BUILD_OPTIONS: return "CL_INVALID_BUILD_OPTIONS";
                case CL_INVALID_PROGRAM: return "CL_INVALID_PROGRAM";
                case CL_INVALID_PROGRAM_EXECUTABLE: return "CL_INVALID_PROGRAM_EXECUTABLE";
                case CL_INVALID_KERNEL_NAME: return "CL_INVALID_KERNEL_NAME";
                case CL_INVALID_KERNEL_DEFINITION: return "CL_INVALID_KERNEL_DEFINITION";
                case CL_INVALID_KERNEL: return "CL_INVALID_KERNEL";
                case CL_INVALID_ARG_INDEX: return "CL_INVALID_ARG_INDEX";
                case CL_INVALID_ARG_VALUE: return "CL_INVALID_ARG_VALUE";
                case CL_INVALID_ARG_SIZE: return "CL_INVALID_ARG_SIZE";
                case CL_INVALID_KERNEL_ARGS: return "CL_INVALID_KERNEL_ARGS";
                case CL_INVALID_WORK_DIMENSION: return "CL_INVALID_WORK_DIMENSION";
                case CL_INVALID_WORK_GROUP_SIZE: return "CL_INVALID_WORK_GROUP_SIZE";
                case CL_INVALID_WORK_ITEM_SIZE: return "CL_INVALID_WORK_ITEM_SIZE";
                case CL_INVALID_GLOBAL_OFFSET: return "CL_INVALID_GLOBAL_OFFSET";
                case CL_INVALID_EVENT_WAIT_LIST: return "CL_INVALID_EVENT_WAIT_LIST";
                case CL_INVALID_EVENT: return "CL_INVALID_EVENT";
                case CL_INVALID_OPERATION: return "CL_INVALID_OPERATION";
                case CL_INVALID_GL_OBJECT: return "CL_INVALID_GL_OBJECT";
                case CL_INVALID_BUFFER_SIZE: return "CL_INVALID_BUFFER_SIZE";
                case CL_INVALID_MIP_LEVEL: return "CL_INVALID_MIP_LEVEL";
                case CL_INVALID_GLOBAL_WORK_SIZE: return "CL_INVALID_GLOBAL_WORK_SIZE";
                case CL_INVALID_PROPERTY: return "CL_INVALID_PROPERTY";
                case CL_INVALID_IMAGE_DESCRIPTOR: return "CL_INVALID_IMAGE_DESCRIPTOR";
                case CL_INVALID_COMPILER_OPTIONS: return "CL_INVALID_COMPILER_OPTIONS";
                case CL_INVALID_LINKER_OPTIONS: return "CL_INVALID_LINKER_OPTIONS";
                case CL_INVALID_DEVICE_PARTITION_COUNT: return "CL_INVALID_DEVICE_PARTITION_COUNT";
                case CL_INVALID_PIPE_SIZE: return "CL_INVALID_PIPE_SIZE";
                case CL_INVALID_DEVICE_QUEUE: return "CL_INVALID_DEVICE_QUEUE";

                default: throw std::runtime_error("Unknown error");
        }
}


const char* cl_device_type_to_string(cl_device_type device_type)
{
        switch (device_type)
        {
                case CL_DEVICE_TYPE_CPU: return "CL_DEVICE_TYPE_CPU";
                case CL_DEVICE_TYPE_GPU: return "CL_DEVICE_TYPE_GPU";
                case CL_DEVICE_TYPE_ACCELERATOR: return "CL_DEVICE_TYPE_ACCELERATOR";
                case CL_DEVICE_TYPE_DEFAULT: return "CL_DEVICE_TYPE_DEFAULT";
                case CL_DEVICE_TYPE_CUSTOM: return "CL_DEVICE_TYPE_CUSTOM";

                default: throw std::runtime_error("Unknown device type");
        }
}


typedef enum
{
        platform = 'p',
        device = 'd',
        help = 'h'
} opt_val;


struct device_info
{
        cl_device_id id;
        std::string name;
};


struct platform_info
{
        cl_platform_id id;
        std::string name;
        std::vector<struct device_info> devices;
};


static void print_available_platforms_and_devices(std::vector<struct platform_info>& platforms)
{
        std::cerr << "Available platforms and their devices:" << std::endl;
        for (size_t i = 0; i < platforms.size(); i++)
        {
                std::cerr << "Platform " << i << ": " << platforms[i].name << std::endl;
                for (size_t j = 0; j < platforms[i].devices.size(); j++)
                {
                        std::cerr << "Device " << j << ": " << platforms[i].devices[j].name << std::endl;
                }
        }
}


static void print_help(std::vector<struct platform_info>& platforms)
{
        std::cerr << "Available parameters:" << std::endl;
        std::cerr << "\t-h, --help\t\t\tprints this help" << std::endl;
        std::cerr << "\t-p, --platform=<platform>\tselects OpenCL platform" << std::endl;
        std::cerr << "\t-d, --device=<device>\t\tselects OpenCL device" << std::endl;

        print_available_platforms_and_devices(platforms);
}


static cl_device_id get_selected_device(struct platform_info& platform,
                                        char* selected_device)
{
        try
        {
                size_t selected_device_index = std::stoul(selected_device);
                if (selected_device_index >= platform.devices.size())
                {
                        return nullptr;
                }
                else
                {
                        return platform.devices[selected_device_index].id;
                }
        }
        catch (std::invalid_argument& e)
        {
                for (size_t i = 0; i < platform.devices.size(); i++)
                {
                        if (platform.devices[i].name == selected_device)
                        {
                                return platform.devices[i].id;
                        }
                }
        }

        return nullptr;
}


static std::string platform_name(cl_platform_id platform_id)
{
        cl_int error = 0;

        size_t name_size = 0;
        error = clGetPlatformInfo(platform_id, CL_PLATFORM_NAME, 0, nullptr, &name_size);
        opencl_check_error(error, clGetPlatformInfo);
        check(name_size != 0);

        char name[name_size];
        error = clGetPlatformInfo(platform_id, CL_PLATFORM_NAME, name_size * sizeof(char), name, nullptr);
        opencl_check_error(error, clGetPlatformInfo);

        return std::string(name);
}


static std::vector<struct device_info> platform_devices(cl_platform_id platform)
{
        cl_int error = 0;

        cl_uint num_devices = 0;
        error = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, nullptr, &num_devices);
        opencl_check_error(error, clGetDeviceIDs);
        check(num_devices != 0);

        cl_device_id device_ids[num_devices];
        std::vector<struct device_info> devices(num_devices);
        error = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, num_devices, device_ids, nullptr);
        opencl_check_error(error, clGetDeviceIDs);

        for (cl_uint i = 0; i < num_devices; i++)
        {
                size_t name_size = 0;
                devices[i].id = device_ids[i];
                error = clGetDeviceInfo(device_ids[i], CL_DEVICE_NAME, 0, nullptr, &name_size);
                opencl_check_error(error, clGetDeviceInfo);
                check(name_size != 0);

                devices[i].name.resize(name_size);
                error = clGetDeviceInfo(device_ids[i], CL_DEVICE_NAME, name_size, &devices[i].name[0], nullptr);
                opencl_check_error(error, clGetDeviceInfo);
        }

        return devices;
}


static std::vector<struct platform_info> opencl_platforms(void)
{
        cl_int error = 0;

        cl_uint num_platforms = 0;
        error = clGetPlatformIDs(0, nullptr, &num_platforms);
        opencl_check_error(error, clGetPlatformIDs);
        check(num_platforms != 0);

        std::vector<cl_platform_id> platform_ids(num_platforms);
        std::vector<struct platform_info> platforms(num_platforms);

        error = clGetPlatformIDs(num_platforms, &platform_ids[0], nullptr);
        opencl_check_error(error, clGetPlatformIDs);

        for (cl_uint i = 0; i < num_platforms; i++)
        {
                platforms[i].id = platform_ids[i];
                platforms[i].name = platform_name(platforms[i].id);
                platforms[i].devices = platform_devices(platforms[i].id);
        }

        return platforms;
}


std::pair<cl_platform_id, cl_device_id> selected_platform_and_device(int argc, char** argv)
{
        const char* optstring = "p:d:h";
        const struct option longopts[] = {
                                                { .name = "platform", .has_arg = required_argument, .flag = nullptr, .val = platform },
                                                { .name = "device", .has_arg = required_argument, .flag = nullptr, .val = device },
                                                { .name = "help", .has_arg = no_argument, .flag = nullptr, .val = help },
                                         };

        std::pair<cl_platform_id, cl_device_id> platform_and_device(nullptr, nullptr);
        std::vector<struct platform_info> platforms = opencl_platforms();

        cl_platform_id selected_platform_id = nullptr;
        cl_device_id selected_device_id = nullptr;

        int getopt_ret_value = 0;
        char* selected_platform = nullptr;
        char* selected_device = nullptr;

        while ((getopt_ret_value = getopt_long(argc, argv, optstring, longopts, nullptr)) != -1)
        {
                switch (getopt_ret_value)
                {
                        case platform:
                        {
                                selected_platform = optarg;
                                break;
                        }
                        case device:
                        {
                                selected_device = optarg;
                                break;
                        }
                        case help:
                        default:
                        {
                                print_help(platforms);
                                return platform_and_device;
                        }
                }
        }

        if (selected_platform == nullptr)
        {
                std::cerr << "You haven't selected platform yet" << std::endl;
                print_help(platforms);
                std::exit(-1);
        }
        if (selected_device == nullptr)
        {
                std::cerr << "You haven't selected device yet" << std::endl;
                print_help(platforms);
                std::exit(-2);
        }

        try
        {
                size_t selected_platform_index = std::stoul(selected_platform);
                selected_platform_id = platforms[selected_platform_index].id;
                selected_device_id = get_selected_device(platforms[selected_platform_index], selected_device);
        }
        catch (std::invalid_argument& e)
        {
                for (size_t i = 0; i < platforms.size(); i++)
                {
                        if (platforms[i].name == selected_platform)
                        {
                                selected_platform_id = platforms[i].id;
                                selected_device_id = get_selected_device(platforms[i], selected_device);
                        }
                }
        }

        if (selected_platform_id == nullptr)
        {
                std::cerr << "Unknown platform: " << selected_platform << std::endl;
                print_available_platforms_and_devices(platforms);
                std::exit(-1);
        }
        if (selected_device_id == nullptr)
        {
                std::cerr << "Unknown device: " << selected_platform << std::endl;
                print_available_platforms_and_devices(platforms);
                std::exit(-1);
        }

        std::cout << "Selected platform " << selected_platform << std::endl;
        std::cout << "Selected device " << selected_device << std::endl;

        platform_and_device.first = selected_platform_id;
        platform_and_device.second = selected_device_id;

        return platform_and_device;
}


opencl_version_t opencl_version(cl_device_id device)
{
        cl_int error = 0;

        size_t device_version_size = 0;
        error = clGetDeviceInfo(device, CL_DEVICE_VERSION, 0, nullptr, &device_version_size);
        opencl_check_error(error, clGetDeviceInfo);
        check(device_version_size != 0);

        char device_version[device_version_size];
        error = clGetDeviceInfo(device, CL_DEVICE_VERSION, device_version_size, device_version, nullptr);
        opencl_check_error(error, clGetDeviceInfo);

        if (strstr(device_version, "OpenCL 1.0") != nullptr)
        {
                return OPENCL_1_0;
        }
        else if (strstr(device_version, "OpenCL 1.1") != nullptr)
        {
                return OPENCL_1_1;
        }
        else if (strstr(device_version, "OpenCL 1.2") != nullptr)
        {
                return OPENCL_1_2;
        }
        else if (strstr(device_version, "OpenCL 2.0") != nullptr)
        {
                return OPENCL_2_0;
        }

        throw std::runtime_error("Unknown OpenCL version");
}



cl_program build_program_from_source(cl_device_id device,
                                     cl_context context,
                                     const char** strings,
                                     size_t* lengths,
                                     const char* options)
{
        cl_int error = 0;

        cl_program program = clCreateProgramWithSource(context, 1, strings, lengths, &error);
        opencl_check_error(error, clCreateProgramWithSource);

        error = clBuildProgram(program, 1, &device, options, nullptr, nullptr);
        if (error != CL_SUCCESS)
        {
                size_t log_size = 0;
                error = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
                opencl_check_error(error, clGetProgramBuildInfo);
                check(log_size != 0);

                char build_log[log_size];
                error = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, build_log, nullptr);
                opencl_check_error(error, clGetProgramBuildInfo);
                std::cerr << "OpenCL driver reports program compilation failure:" << std::endl;
                std::cerr << build_log << std::endl;
                throw std::runtime_error("");
        }

        return program;
}
