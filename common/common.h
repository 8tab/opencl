#pragma once

#include <CL/cl.h>
#include <utility>
#include <sstream>
#include <stdexcept>
#include <cstdio>


typedef enum
{
        OPENCL_1_0 = 0,
        OPENCL_1_1,
        OPENCL_1_2,
        OPENCL_2_0
} opencl_version_t;

const char* opencl_error_to_string(cl_int error);

const char* cl_device_type_to_string(cl_device_type device_type);

std::pair<cl_platform_id, cl_device_id> selected_platform_and_device(int argc, char** argv);

opencl_version_t opencl_version(cl_device_id device);

cl_program build_program_from_source(cl_device_id device,
                                     cl_context context,
                                     const char** strings,
                                     size_t* lengths,
                                     const char* options=nullptr);


#define NL "\n"


#define opencl_check_error(err_code, function) \
do \
{ \
        if ((err_code)) \
        { \
                std::stringstream msg; \
                msg << "[" << __FILE__ << ":" << __LINE__ << "] " << #function \
                    << " returns " << opencl_error_to_string((err_code)); \
                throw std::runtime_error(msg.str()); \
        } \
} \
while (false)


#define check(condition) \
do \
{ \
        if (!(condition)) \
        { \
                std::stringstream msg; \
                msg << "[" << __FILE__ << ":" << __LINE__ + "] '" << #condition << "' is not met"; \
                throw std::runtime_error(msg.str()); \
        } \
} \
while (false)


#define cl_wrapper(type, release_function) \
class type##_wrapper \
{ \
        public: \
                type##_wrapper() : _entity(nullptr) \
                { \
                } \
\
                type##_wrapper(type& entity) : _entity(entity) \
                { \
                } \
\
                type##_wrapper& operator=(type& entity) \
                { \
                        _entity = entity; \
                        return *this; \
                } \
\
                type##_wrapper& operator=(type##_wrapper& entity) \
                { \
                        _entity = entity; \
                        return *this; \
                } \
\
                operator type() \
                { \
                        return _entity; \
                } \
\
                ~type##_wrapper() \
                { \
                        if (_entity) \
                        { \
                                cl_int err = release_function(_entity); \
                                opencl_check_error(err, release_function); \
                        } \
                } \
\
        private: \
                type _entity; \
}; \


cl_wrapper(cl_context, clReleaseContext);
cl_wrapper(cl_command_queue, clReleaseCommandQueue);
cl_wrapper(cl_program, clReleaseProgram);
cl_wrapper(cl_kernel, clReleaseKernel);

