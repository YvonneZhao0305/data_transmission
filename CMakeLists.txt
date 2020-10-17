find_package(CUDA)
find_package(OpenCV REQUIRED)

FILE(GLOB fuc ../include/fuc.cpp ../include/fuc.h)
include_directories ("${PROJECT_SOURCE_DIR}")
include_directories(${CUDA_INCLUDE_DIRS})

set(CUDA_NVCC_FLAGS -arch=compute_30;-G;-g)

CUDA_ADD_EXECUTABLE(data_transmission_test data_transmission_test.cu ${fuc})
target_link_libraries(data_transmission_test ${OpenCV_LIBS})