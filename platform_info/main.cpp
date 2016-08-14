#include <iostream>
#include <map>
#include "common.h"


void print_platform_info(cl_platform_id platform_id);


int main(void)
{
        cl_int error = 0;

        cl_uint num_platforms = 0;
        error = clGetPlatformIDs(0, nullptr, &num_platforms);
        opencl_check_error(error, clGetPlatformIDs);
        check(num_platforms != 0);

        cl_platform_id platforms[num_platforms];
        error = clGetPlatformIDs(num_platforms, platforms, nullptr);
        opencl_check_error(error, clGetPlatformIDs);

        for (cl_uint i = 0; i < num_platforms; i++)
        {
                std::cout << "Platform " << i << " :" << std::endl;
                print_platform_info(platforms[i]);
                std::cout << std::endl << std::endl;
        }

        return 0;
}


void print_device_type(cl_device_id device_id)
{
        cl_int error = 0;
        cl_device_type device_type = 0;
        error = clGetDeviceInfo(device_id, CL_DEVICE_TYPE, sizeof(cl_device_type), &device_type, nullptr);
        opencl_check_error(error, clGetDeviceInfo);
        std::cout << "\t\tCL_DEVICE_TYPE: " << cl_device_type_to_string(device_type) << std::endl;
}


template <typename T>
void print_device_param(cl_device_id device_id, cl_device_info device_info, const char* device_info_name)
{
        cl_int error = 0;
        T param = 0;
        error = clGetDeviceInfo(device_id, device_info, sizeof(T), &param, nullptr);
        opencl_check_error(error, clGetDeviceInfo);
        std::cout << "\t\t" << device_info_name << ": " << param << std::endl;
}


void print_device_max_work_item_dim_and_sizes(cl_device_id device_id)
{
        cl_int error = 0;
        cl_uint max_work_item_dimensions = 0;
        error = clGetDeviceInfo(device_id, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(cl_uint), &max_work_item_dimensions, nullptr);
        opencl_check_error(error, clGetDeviceInfo);
        check(max_work_item_dimensions != 0);
        std::cout << "\t\tCL_DEVICE_MAX_WORK_ITEM_DIMENSIONS: " << max_work_item_dimensions << std::endl;

        size_t max_work_item_sizes[max_work_item_dimensions];
        error = clGetDeviceInfo(device_id, CL_DEVICE_MAX_WORK_ITEM_SIZES, max_work_item_dimensions * sizeof(size_t), max_work_item_sizes, nullptr);
        opencl_check_error(error, clGetDeviceInfo);
        std::cout << "\t\tCL_DEVICE_MAX_WORK_ITEM_SIZES: {";
        for (size_t i = 0; i < max_work_item_dimensions; i++)
        {
                std::cout << max_work_item_sizes[i];
                if (i < (max_work_item_dimensions - 1))
                {
                        std::cout << ",";
                }
        }
        std::cout << "}" << std::endl;
}


void print_device_fp_config(cl_device_id device_id)
{
        cl_int error = 0;
        cl_device_fp_config single_fp_config;
        error = clGetDeviceInfo(device_id, CL_DEVICE_SINGLE_FP_CONFIG, sizeof(cl_device_fp_config), &single_fp_config, nullptr);
        opencl_check_error(error, clGetDeviceInfo);

        std::map<int, const char*> fp_config_name;
        fp_config_name[CL_FP_DENORM] = "CL_FP_DENORM";
        fp_config_name[CL_FP_INF_NAN] = "CL_FP_INF_NAN";
        fp_config_name[CL_FP_ROUND_TO_NEAREST] = "CL_FP_ROUND_TO_NEAREST";
        fp_config_name[CL_FP_ROUND_TO_ZERO] = "CL_FP_ROUND_TO_ZERO";
        fp_config_name[CL_FP_ROUND_TO_INF] = "CL_FP_ROUND_TO_INF";
        fp_config_name[CL_FP_FMA] = "CL_FP_FMA";
        fp_config_name[CL_FP_SOFT_FLOAT] = "CL_FP_SOFT_FLOAT";
        fp_config_name[CL_FP_CORRECTLY_ROUNDED_DIVIDE_SQRT] = "CL_FP_CORRECTLY_ROUNDED_DIVIDE_SQRT";

        std::cout << "\t\tCL_DEVICE_SINGLE_FP_CONFIG: ";
        for (auto config : fp_config_name)
        {
                if ((single_fp_config & config.first) != 0)
                {
                        std::cout << config.second << "|";
                }
        }
        std::cout << std::endl;

        cl_device_fp_config double_fp_config;
        error = clGetDeviceInfo(device_id, CL_DEVICE_DOUBLE_FP_CONFIG, sizeof(cl_device_fp_config), &double_fp_config, nullptr);
        opencl_check_error(error, clGetDeviceInfo);

        std::cout << "\t\tCL_DEVICE_DOUBLE_FP_CONFIG: ";
        for (auto config : fp_config_name)
        {
                if ((double_fp_config & config.first) != 0)
                {
                        std::cout << config.second << "|";
                }
        }
        std::cout << std::endl;
}


void print_device_global_mem_cache(cl_device_id device_id)
{
        cl_int error = 0;
        cl_device_mem_cache_type mem_cache_type = 0;
        error = clGetDeviceInfo(device_id, CL_DEVICE_GLOBAL_MEM_CACHE_TYPE, sizeof(cl_device_mem_cache_type), &mem_cache_type, nullptr);
        opencl_check_error(error, clGetDeviceInfo);

        std::cout << "\t\tCL_DEVICE_GLOBAL_MEM_CACHE_TYPE: ";
        switch (mem_cache_type)
        {
                case CL_NONE: std::cout << "CL_NONE" << std::endl; break;
                case CL_READ_ONLY_CACHE: std::cout << "CL_READ_ONLY_CACHE" << std::endl; break;
                case CL_READ_WRITE_CACHE: std::cout << "CL_READ_WRITE_CACHE" << std::endl; break;
                default: std::runtime_error("Unknown device memory cache type");
        }
}


void print_device_local_mem_type(cl_device_id device_id)
{
        cl_int error = 0;
        cl_device_local_mem_type local_mem_type;
        error = clGetDeviceInfo(device_id, CL_DEVICE_LOCAL_MEM_TYPE, sizeof(cl_device_local_mem_type), &local_mem_type, nullptr);
        opencl_check_error(error, clGetDeviceInfo);

        std::cout << "\t\tCL_DEVICE_LOCAL_MEM_TYPE: ";
        switch (local_mem_type)
        {
                case CL_NONE: std::cout << "CL_NONE" << std::endl; break;
                case CL_LOCAL: std::cout << "CL_LOCAL" << std::endl; break;
                case CL_GLOBAL: std::cout << "CL_GLOBAL" << std::endl; break;
                default: std::runtime_error("Unknown device local memory type");
        }
}


void print_device_execution_capabilities(cl_device_id device_id)
{
        cl_int error = 0;

        cl_device_exec_capabilities device_exec_caps;
        error = clGetDeviceInfo(device_id, CL_DEVICE_EXECUTION_CAPABILITIES, sizeof(device_exec_caps), &device_exec_caps, nullptr);
        opencl_check_error(error, clGetDeviceInfo);

        std::cout << "\t\tCL_DEVICE_EXECUTION_CAPABILITIES: ";
        if (device_exec_caps & CL_EXEC_KERNEL)
        {
                std::cout << "CL_EXEC_KERNEL|";
        }
        else
        {
                throw std::runtime_error("ERROR: CL_EXEC_KERNEL is not present in CL_DEVICE_EXECUTION_CAPABILITIES reported by the driver");
        }
        if (device_exec_caps & CL_EXEC_NATIVE_KERNEL)
        {
                std::cout << "CL_EXEC_NATIVE_KERNEL|";
        }
        std::cout << std::endl;
}


void print_device_queue_properties(cl_device_id device_id, cl_device_info param_name)
{
        cl_int error = 0;

        cl_command_queue_properties cmd_queue_props;
        error = clGetDeviceInfo(device_id, param_name, sizeof(cmd_queue_props), &cmd_queue_props, nullptr);
        opencl_check_error(error, clGetDeviceInfo);

        if (param_name == CL_DEVICE_QUEUE_ON_HOST_PROPERTIES)
        {
                std::cout << "\t\tCL_DEVICE_QUEUE_ON_HOST_PROPERTIES: ";
        }
#if CL_VERSION_2_0 == 1
        else if (param_name == CL_DEVICE_QUEUE_ON_DEVICE_PROPERTIES)
        {
                std::cout << "\t\tCL_DEVICE_QUEUE_ON_DEVICE_PROPERTIES: ";
        }
#endif
        else
        {
                throw std::runtime_error("Unknown param_name for clGetDeviceInfo. Should be CL_DEVICE_QUEUE_ON_HOST_PROPERTIES or CL_DEVICE_QUEUE_ON_DEVICE_PROPERTIES");
        }
        if (cmd_queue_props & CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE)
        {
                std::cout << "CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE|";
        }
        if (cmd_queue_props & CL_QUEUE_PROFILING_ENABLE)
        {
                std::cout << "CL_QUEUE_PROFILING_ENABLE|";
        }
        else
        {
                throw std::runtime_error("ERROR: CL_QUEUE_PROFILING_ENABLE is not preset in CL_DEVICE_QUEUE_ON_HOST_PROPERTIES reported by the driver");
        }
        std::cout << std::endl;
}

#if CL_VERSION_1_2 == 1
void print_device_built_in_kernels(cl_device_id device_id)
{
        cl_int error = 0;

        size_t built_in_kernels_size = 0;
        error = clGetDeviceInfo(device_id, CL_DEVICE_BUILT_IN_KERNELS, 0, NULL, &built_in_kernels_size);
        opencl_check_error(error, clGetDeviceInfo);
        check(built_in_kernels_size != 0);

        char built_in_kernels[built_in_kernels_size];
        error = clGetDeviceInfo(device_id, CL_DEVICE_BUILT_IN_KERNELS, built_in_kernels_size, built_in_kernels, nullptr);
        opencl_check_error(error, clGetDeviceInfo);

        std::cout << "\t\tCL_DEVICE_BUILT_IN_KERNELS: " << built_in_kernels << std::endl;
}
#endif


void print_device_string(cl_device_id device_id, cl_device_info param_name)
{
        cl_int error = 0;

        size_t param_size = 0;
        error = clGetDeviceInfo(device_id, param_name, 0, nullptr, &param_size);
        opencl_check_error(error, clGetDeviceInfo);
        check(param_size != 0);

        char param_value[param_size];
        error = clGetDeviceInfo(device_id, param_name, param_size, param_value, nullptr);
        opencl_check_error(error, clGetDeviceInfo);

        std::map<cl_device_info, std::string> param_name_map;
        param_name_map[CL_DEVICE_NAME] = "CL_DEVICE_NAME";
        param_name_map[CL_DEVICE_VENDOR] = "CL_DEVICE_VENDOR";
        param_name_map[CL_DRIVER_VERSION] = "CL_DRIVER_VERSION";
        param_name_map[CL_DEVICE_PROFILE] = "CL_DEVICE_PROFILE";
        param_name_map[CL_DEVICE_VERSION] = "CL_DEVICE_VERSION";
        param_name_map[CL_DEVICE_OPENCL_C_VERSION] = "CL_DEVICE_OPENCL_C_VERSION";
        param_name_map[CL_DEVICE_EXTENSIONS] = "CL_DEVICE_EXTENSIONS";

        if (param_name_map.find(param_name) != param_name_map.end())
        {
                std::cout << "\t\t" << param_name_map[param_name] << ": ";
        }
        else
        {
                throw std::runtime_error("print_device_string: unsupported param_name");
        }
        std::cout << param_value << std::endl;
}


void print_device_partition_properties(cl_device_id device_id)
{
        cl_int error = 0;

        size_t param_size = 0;
        error = clGetDeviceInfo(device_id, CL_DEVICE_PARTITION_PROPERTIES, 0, nullptr, &param_size);
        opencl_check_error(error, clGetDeviceInfo);
        check(param_size != 0);

        size_t n = param_size / sizeof(cl_device_partition_property);
        cl_device_partition_property properties[n];
        error = clGetDeviceInfo(device_id, CL_DEVICE_PARTITION_PROPERTIES, param_size, properties, nullptr);
        opencl_check_error(error, clGetDeviceInfo);

        std::cout << "\t\tCL_DEVICE_PARTITION_PROPERTIES: ";
        for (size_t i = 0; i < n; i++)
        {
                switch (properties[i])
                {
                        case CL_DEVICE_PARTITION_EQUALLY:
                        {
                                std::cout << "CL_DEVICE_PARTITION_EQUALLY|";
                                break;
                        }
                        case CL_DEVICE_PARTITION_BY_COUNTS:
                        {
                                std::cout << "CL_DEVICE_PARTITION_BY_COUNTS|";
                                break;
                        }
                        case CL_DEVICE_PARTITION_BY_AFFINITY_DOMAIN:
                        {
                                std::cout << "CL_DEVICE_PARTITION_BY_AFFINITY_DOMAIN|";
                                break;
                        }
                }
        }
        std::cout << std::endl;
}


void print_device_partition_affinity_domain(cl_device_id device_id)
{
        cl_int error = 0;

        size_t param_size = 0;
        error = clGetDeviceInfo(device_id, CL_DEVICE_PARTITION_AFFINITY_DOMAIN, 0, nullptr, &param_size);
        opencl_check_error(error, clGetDeviceInfo);
        check(param_size != 0);

        cl_device_affinity_domain domain;
        error = clGetDeviceInfo(device_id, CL_DEVICE_PARTITION_AFFINITY_DOMAIN, param_size, &domain, nullptr);
        opencl_check_error(error, clGetDeviceInfo);

        std::cout << "\t\tCL_DEVICE_PARTITION_AFFINITY_DOMAIN: ";
        if (domain & CL_DEVICE_AFFINITY_DOMAIN_NUMA)
        {
                std::cout << "CL_DEVICE_AFFINITY_DOMAIN_NUMA|";
        }
        if (domain & CL_DEVICE_AFFINITY_DOMAIN_L4_CACHE)
        {
                std::cout << "CL_DEVICE_AFFINITY_DOMAIN_L4_CACHE|";
        }
        if (domain & CL_DEVICE_AFFINITY_DOMAIN_L3_CACHE)
        {
                std::cout << "CL_DEVICE_AFFINITY_DOMAIN_L3_CACHE|";
        }
        if (domain & CL_DEVICE_AFFINITY_DOMAIN_L2_CACHE)
        {
                std::cout << "CL_DEVICE_AFFINITY_DOMAIN_L2_CACHE|";
        }
        if (domain & CL_DEVICE_AFFINITY_DOMAIN_L1_CACHE)
        {
                std::cout << "CL_DEVICE_AFFINITY_DOMAIN_L1_CACHE|";
        }
        if (domain & CL_DEVICE_AFFINITY_DOMAIN_NEXT_PARTITIONABLE)
        {
                std::cout << "CL_DEVICE_AFFINITY_DOMAIN_NEXT_PARTITIONABLE|";
        }
        std::cout << std::endl;
}


void print_device_partition_type(cl_device_id device_id)
{
        cl_int error = 0;

        std::cout << "\t\tCL_DEVICE_PARTITION_TYPE: ";

        size_t param_size = 0;
        error = clGetDeviceInfo(device_id, CL_DEVICE_PARTITION_TYPE, 0, nullptr, &param_size);
        opencl_check_error(error, clGetDeviceInfo);
        if (param_size == 0)
        {
                std::cout << "no partition type associated with device" << std::endl;
                return;
        }

        size_t n = param_size / sizeof(cl_device_partition_property);
        cl_device_partition_property properties[n];
        error = clGetDeviceInfo(device_id, CL_DEVICE_PARTITION_TYPE, param_size, properties, nullptr);
        opencl_check_error(error, clGetDeviceInfo);

        for (size_t i = 0; i < n; i++)
        {
                switch (properties[i])
                {
                        case CL_DEVICE_PARTITION_BY_AFFINITY_DOMAIN:
                        {
                                std::cout << "CL_DEVICE_PARTITION_BY_AFFINITY_DOMAIN|";
                                break;
                        }
                        case CL_DEVICE_AFFINITY_DOMAIN_NEXT_PARTITIONABLE:
                        {
                                std::cout << "CL_DEVICE_AFFINITY_DOMAIN_NEXT_PARTITIONABLE|";
                                break;
                        }
                }
        }
        std::cout << std::endl;
}


void print_device_svm_capabilities(cl_device_id device_id)
{
        cl_int error = 0;

        cl_device_svm_capabilities svm_caps;
        error = clGetDeviceInfo(device_id, CL_DEVICE_SVM_CAPABILITIES, sizeof(cl_device_svm_capabilities), &svm_caps, nullptr);
        opencl_check_error(error, clGetDeviceInfo);

        std::cout << "\t\tCL_DEVICE_PARTITION_TYPE: ";
        switch (svm_caps)
        {
                case CL_DEVICE_SVM_COARSE_GRAIN_BUFFER:
                {
                        std::cout << "CL_DEVICE_SVM_COARSE_GRAIN_BUFFER|";
                        break;
                }
                case CL_DEVICE_SVM_FINE_GRAIN_BUFFER:
                {
                        std::cout << "CL_DEVICE_SVM_FINE_GRAIN_BUFFER|";
                        break;
                }
                case CL_DEVICE_SVM_FINE_GRAIN_SYSTEM:
                {
                        std::cout << "CL_DEVICE_SVM_FINE_GRAIN_SYSTEM|";
                        break;
                }
                case CL_DEVICE_SVM_ATOMICS:
                {
                        std::cout << "CL_DEVICE_SVM_ATOMICS|";
                        break;
                }
        }
        std::cout << std::endl;
}


void print_device_info(cl_device_id device_id)
{
        opencl_version_t device_opencl_version = opencl_version(device_id);
        print_device_type(device_id); // CL_DEVICE_TYPE
        print_device_param<cl_uint>(device_id, CL_DEVICE_VENDOR_ID, "CL_DEVICE_VENDOR_ID");
        print_device_param<cl_uint>(device_id, CL_DEVICE_MAX_COMPUTE_UNITS, "CL_DEVICE_MAX_COMPUTE_UNITS");
        print_device_max_work_item_dim_and_sizes(device_id); // CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, CL_DEVICE_MAX_WORK_ITEM_SIZES
        print_device_param<size_t>(device_id, CL_DEVICE_MAX_WORK_GROUP_SIZE, "CL_DEVICE_MAX_WORK_GROUP_SIZE");
        print_device_param<cl_uint>(device_id, CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR, "CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR");
        print_device_param<cl_uint>(device_id, CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT, "CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT");
        print_device_param<cl_uint>(device_id, CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT, "CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT");
        print_device_param<cl_uint>(device_id, CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG, "CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG");
        print_device_param<cl_uint>(device_id, CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT, "CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT");
        print_device_param<cl_uint>(device_id, CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE, "CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE");
        print_device_param<cl_uint>(device_id, CL_DEVICE_PREFERRED_VECTOR_WIDTH_HALF, "CL_DEVICE_PREFERRED_VECTOR_WIDTH_HALF");
        print_device_param<cl_uint>(device_id, CL_DEVICE_NATIVE_VECTOR_WIDTH_CHAR, "CL_DEVICE_NATIVE_VECTOR_WIDTH_CHAR");
        print_device_param<cl_uint>(device_id, CL_DEVICE_NATIVE_VECTOR_WIDTH_SHORT, "CL_DEVICE_NATIVE_VECTOR_WIDTH_SHORT");
        print_device_param<cl_uint>(device_id, CL_DEVICE_NATIVE_VECTOR_WIDTH_INT, "CL_DEVICE_NATIVE_VECTOR_WIDTH_INT");
        print_device_param<cl_uint>(device_id, CL_DEVICE_NATIVE_VECTOR_WIDTH_LONG, "CL_DEVICE_NATIVE_VECTOR_WIDTH_LONG");
        print_device_param<cl_uint>(device_id, CL_DEVICE_NATIVE_VECTOR_WIDTH_FLOAT, "CL_DEVICE_NATIVE_VECTOR_WIDTH_FLOAT");
        print_device_param<cl_uint>(device_id, CL_DEVICE_NATIVE_VECTOR_WIDTH_DOUBLE, "CL_DEVICE_NATIVE_VECTOR_WIDTH_DOUBLE");
        print_device_param<cl_uint>(device_id, CL_DEVICE_NATIVE_VECTOR_WIDTH_HALF, "CL_DEVICE_NATIVE_VECTOR_WIDTH_HALF");
        print_device_param<cl_uint>(device_id, CL_DEVICE_MAX_CLOCK_FREQUENCY, "CL_DEVICE_MAX_CLOCK_FREQUENCY");
        print_device_param<cl_uint>(device_id, CL_DEVICE_ADDRESS_BITS, "CL_DEVICE_ADDRESS_BITS");
        print_device_param<cl_ulong>(device_id, CL_DEVICE_MAX_MEM_ALLOC_SIZE, "CL_DEVICE_MAX_MEM_ALLOC_SIZE");
        print_device_param<cl_bool>(device_id, CL_DEVICE_IMAGE_SUPPORT, "CL_DEVICE_IMAGE_SUPPORT");
        print_device_param<cl_uint>(device_id, CL_DEVICE_MAX_READ_IMAGE_ARGS, "CL_DEVICE_MAX_READ_IMAGE_ARGS");
        print_device_param<cl_uint>(device_id, CL_DEVICE_MAX_WRITE_IMAGE_ARGS, "CL_DEVICE_MAX_WRITE_IMAGE_ARGS");
        print_device_param<size_t>(device_id, CL_DEVICE_IMAGE2D_MAX_WIDTH, "CL_DEVICE_IMAGE2D_MAX_WIDTH");
        print_device_param<size_t>(device_id, CL_DEVICE_IMAGE2D_MAX_HEIGHT, "CL_DEVICE_IMAGE2D_MAX_HEIGHT");
        print_device_param<size_t>(device_id, CL_DEVICE_IMAGE3D_MAX_WIDTH, "CL_DEVICE_IMAGE3D_MAX_WIDTH");
        print_device_param<size_t>(device_id, CL_DEVICE_IMAGE3D_MAX_HEIGHT, "CL_DEVICE_IMAGE3D_MAX_HEIGHT");
        print_device_param<size_t>(device_id, CL_DEVICE_IMAGE3D_MAX_DEPTH, "CL_DEVICE_IMAGE3D_MAX_DEPTH");
        print_device_param<size_t>(device_id, CL_DEVICE_IMAGE_MAX_BUFFER_SIZE, "CL_DEVICE_IMAGE_MAX_BUFFER_SIZE");
        print_device_param<size_t>(device_id, CL_DEVICE_IMAGE_MAX_ARRAY_SIZE, "CL_DEVICE_IMAGE_MAX_ARRAY_SIZE");
        print_device_param<cl_uint>(device_id, CL_DEVICE_MAX_SAMPLERS, "CL_DEVICE_MAX_SAMPLERS");
        print_device_param<size_t>(device_id, CL_DEVICE_MAX_PARAMETER_SIZE, "CL_DEVICE_MAX_PARAMETER_SIZE");
        print_device_param<cl_uint>(device_id, CL_DEVICE_MEM_BASE_ADDR_ALIGN, "CL_DEVICE_MEM_BASE_ADDR_ALIGN");
        print_device_fp_config(device_id); // CL_DEVICE_SINGLE_FP_CONFIG, CL_DEVICE_DOUBLE_FP_CONFIG
        print_device_global_mem_cache(device_id); // CL_DEVICE_GLOBAL_MEM_CACHE
        print_device_param<cl_uint>(device_id, CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE, "CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE");
        print_device_param<cl_ulong>(device_id, CL_DEVICE_GLOBAL_MEM_CACHE_SIZE, "CL_DEVICE_GLOBAL_MEM_CACHE_SIZE");
        print_device_param<cl_ulong>(device_id, CL_DEVICE_GLOBAL_MEM_SIZE, "CL_DEVICE_GLOBAL_MEM_SIZE");
        print_device_param<cl_ulong>(device_id, CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE, "CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE");
        print_device_param<cl_uint>(device_id, CL_DEVICE_MAX_CONSTANT_ARGS, "CL_DEVICE_MAX_CONSTANT_ARGS");
        print_device_local_mem_type(device_id); // CL_DEVICE_LOCAL_MEM_TYPE
        print_device_param<cl_ulong>(device_id, CL_DEVICE_LOCAL_MEM_SIZE, "CL_DEVICE_LOCAL_MEM_SIZE");
        print_device_param<cl_bool>(device_id, CL_DEVICE_ERROR_CORRECTION_SUPPORT, "CL_DEVICE_ERROR_CORRECTION_SUPPORT");
        print_device_param<cl_bool>(device_id, CL_DEVICE_HOST_UNIFIED_MEMORY, "CL_DEVICE_HOST_UNIFIED_MEMORY");
        print_device_param<size_t>(device_id, CL_DEVICE_PROFILING_TIMER_RESOLUTION, "CL_DEVICE_PROFILING_TIMER_RESOLUTION");
        print_device_param<cl_bool>(device_id, CL_DEVICE_ENDIAN_LITTLE, "CL_DEVICE_ENDIAN_LITTLE");
        print_device_param<cl_bool>(device_id, CL_DEVICE_AVAILABLE, "CL_DEVICE_AVAILABLE");
        print_device_param<cl_bool>(device_id, CL_DEVICE_COMPILER_AVAILABLE, "CL_DEVICE_COMPILER_AVAILABLE");
#if CL_VERSION_1_2 == 1
        if (device_opencl_version >= OPENCL_1_2)
        {
                print_device_param<cl_bool>(device_id, CL_DEVICE_LINKER_AVAILABLE, "CL_DEVICE_LINKER_AVAILABLE");
        }
#endif
        print_device_execution_capabilities(device_id);
        print_device_queue_properties(device_id, CL_DEVICE_QUEUE_ON_HOST_PROPERTIES);
#if CL_VERSION_2_0 == 1
        if (device_opencl_version >= OPENCL_2_0)
        {
                print_device_queue_properties(device_id, CL_DEVICE_QUEUE_ON_DEVICE_PROPERTIES);
                print_device_param<cl_uint>(device_id, CL_DEVICE_QUEUE_ON_DEVICE_PREFERRED_SIZE, "CL_DEVICE_QUEUE_ON_DEVICE_PREFERRED_SIZE");
                print_device_param<cl_uint>(device_id, CL_DEVICE_QUEUE_ON_DEVICE_MAX_SIZE, "CL_DEVICE_QUEUE_ON_DEVICE_MAX_SIZE");
                print_device_param<cl_uint>(device_id, CL_DEVICE_MAX_ON_DEVICE_EVENTS, "CL_DEVICE_MAX_ON_DEVICE_EVENTS");
        }
#endif
#if CL_VERSION_1_2 == 1
        if (device_opencl_version >= OPENCL_1_2)
        {
                print_device_built_in_kernels(device_id);
        }
#endif
        print_device_param<cl_platform_id>(device_id, CL_DEVICE_PLATFORM, "CL_DEVICE_PLATFORM");
        print_device_string(device_id, CL_DEVICE_NAME);
        print_device_string(device_id, CL_DEVICE_VENDOR);
        print_device_string(device_id, CL_DRIVER_VERSION);
        print_device_string(device_id, CL_DEVICE_PROFILE);
        print_device_string(device_id, CL_DEVICE_VERSION);
        print_device_string(device_id, CL_DEVICE_OPENCL_C_VERSION);
        print_device_string(device_id, CL_DEVICE_EXTENSIONS);
#if CL_VERSION_1_2 == 1
        if (device_opencl_version >= OPENCL_1_2)
        {
                print_device_param<size_t>(device_id, CL_DEVICE_PRINTF_BUFFER_SIZE, "CL_DEVICE_PRINTF_BUFFER_SIZE");
                print_device_param<cl_bool>(device_id, CL_DEVICE_PREFERRED_INTEROP_USER_SYNC, "CL_DEVICE_PREFERRED_INTEROP_USER_SYNC");
                print_device_param<cl_device_id>(device_id, CL_DEVICE_PARENT_DEVICE, "CL_DEVICE_PARENT_DEVICE");
                print_device_param<cl_uint>(device_id, CL_DEVICE_PARTITION_MAX_SUB_DEVICES, "CL_DEVICE_PARTITION_MAX_SUB_DEVICES");
                print_device_partition_properties(device_id); // CL_DEVICE_PARTITION_PROPERTIES
                print_device_partition_affinity_domain(device_id); // CL_DEVICE_PARTITION_AFFINITY_DOMAIN
                print_device_partition_type(device_id); // CL_DEVICE_PARTITION_TYPE
                print_device_param<cl_uint>(device_id, CL_DEVICE_REFERENCE_COUNT, "CL_DEVICE_REFERENCE_COUNT");
        }
#endif
#if CL_VERSION_2_0 == 1
        if (device_opencl_version >= OPENCL_2_0)
        {
                print_device_svm_capabilities(device_id);
                print_device_param<cl_uint>(device_id, CL_DEVICE_PREFERRED_PLATFORM_ATOMIC_ALIGNMENT, "CL_DEVICE_PREFERRED_PLATFORM_ATOMIC_ALIGNMENT");
                print_device_param<cl_uint>(device_id, CL_DEVICE_PREFERRED_GLOBAL_ATOMIC_ALIGNMENT, "CL_DEVICE_PREFERRED_GLOBAL_ATOMIC_ALIGNMENT");
                print_device_param<cl_uint>(device_id, CL_DEVICE_PREFERRED_LOCAL_ATOMIC_ALIGNMENT, "CL_DEVICE_PREFERRED_LOCAL_ATOMIC_ALIGNMENT");
        }
#endif
}


void print_platform_param_info(cl_platform_id platform, cl_platform_info param_name, const char* param_header)
{
        cl_int error = 0;

        size_t param_size = 0;
        error = clGetPlatformInfo(platform, param_name,
                                  0, nullptr, &param_size);
        opencl_check_error(error, clGetPlatformInfo);
        check(param_size != 0);

        char param_value[param_size];
        error = clGetPlatformInfo(platform, param_name,
                                  param_size, param_value, nullptr);
        opencl_check_error(error, clGetPlatformInfo);

        std::cout << "\t" << param_header << ": " << param_value << std::endl;
}


void print_platform_info(cl_platform_id platform_id)
{
        cl_int error = 0;

        print_platform_param_info(platform_id, CL_PLATFORM_PROFILE, "Profile");
        print_platform_param_info(platform_id, CL_PLATFORM_VERSION, "Version");
        print_platform_param_info(platform_id, CL_PLATFORM_NAME, "Name");
        print_platform_param_info(platform_id, CL_PLATFORM_VENDOR, "Vendor");
        print_platform_param_info(platform_id, CL_PLATFORM_EXTENSIONS, "Extensions");

        cl_uint num_devices = 0;
        error = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_ALL, 0, nullptr, &num_devices);
        opencl_check_error(error, clGetDeviceIDs);
        check(num_devices != 0);

        cl_device_id devices[num_devices];
        error = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_ALL, num_devices, devices, nullptr);
        opencl_check_error(error, clGetDeviceIDs);

        std::cout << "\tDevice count: " << num_devices << std::endl << std::endl;
        for (cl_uint i = 0; i < num_devices; i++)
        {
                std::cout << "\tDevice " << i << ":" << std::endl;
                print_device_info(devices[i]);
        }
}

