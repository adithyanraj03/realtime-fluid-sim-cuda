cmake_minimum_required(VERSION 3.18)

# allow unsupported CUDA host compilers (MSVC 2026 / v19.50+)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -allow-unsupported-compiler")

project(NavierStokesFluidSim LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---- CUDA (optional) ----
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    add_definitions(-DUSE_CUDA)
    message(STATUS "CUDA found (${CMAKE_CUDA_COMPILER_VERSION}) - GPU acceleration enabled")
else()
    message(STATUS "CUDA not found - CPU-only build")
endif()

# ---- OpenMP (optional) ----
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP found - parallel CPU ops enabled")
endif()

# ---- GLFW (fetched automatically) ----
include(FetchContent)
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
    glfw
    GIT_REPOSITORY https://github.com/glfw/glfw.git
    GIT_TAG        3.4
)
FetchContent_MakeAvailable(glfw)

# ---- glad (vendored) ----
add_library(glad STATIC
    ${CMAKE_SOURCE_DIR}/extern/glad/src/gl.c
)
target_include_directories(glad PUBLIC ${CMAKE_SOURCE_DIR}/extern/glad/include)

# ---- source files ----
set(COMMON_SOURCES
    src/Grid.cpp
    src/FluidSolver.cpp
    src/Renderer.cpp
    src/Simulation.cpp
)

set(MAIN_SOURCES
    src/main.cpp
)

# ---- executable ----
if(CMAKE_CUDA_COMPILER)
    add_executable(fluid_sim
        ${COMMON_SOURCES}
        ${MAIN_SOURCES}
        src/cuda_kernels.cu
    )
    set_target_properties(fluid_sim PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES "80;86;89;120"
    )
else()
    add_executable(fluid_sim
        ${COMMON_SOURCES}
        ${MAIN_SOURCES}
    )
endif()

target_include_directories(fluid_sim PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(fluid_sim PRIVATE glfw glad)

# platform-specific OpenGL linking
if(WIN32)
    target_link_libraries(fluid_sim PRIVATE opengl32)
elseif(APPLE)
    find_library(OPENGL_LIB OpenGL)
    target_link_libraries(fluid_sim PRIVATE ${OPENGL_LIB})
else()
    target_link_libraries(fluid_sim PRIVATE GL dl)
endif()

if(OpenMP_CXX_FOUND)
    target_link_libraries(fluid_sim PRIVATE OpenMP::OpenMP_CXX)
endif()

# MSVC optimizations (C/C++ only, not CUDA)
if(MSVC)
    target_compile_options(fluid_sim PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/O2 /fp:fast>)
    target_compile_options(fluid_sim PRIVATE $<$<COMPILE_LANGUAGE:C>:/O2 /fp:fast>)
endif()
