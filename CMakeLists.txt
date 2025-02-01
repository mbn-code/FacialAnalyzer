cmake_minimum_required(VERSION 3.10)
project(faceRec VERSION 0.1.0 LANGUAGES C CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# If needed for Apple Silicon, explicitly set architecture
if(APPLE)
    set(CMAKE_OSX_ARCHITECTURES "arm64")
endif()

# Find OpenCV package installed via Homebrew.
# You may need to specify OpenCV_DIR if CMake does not find it automatically.
find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(faceRec main.cpp)

target_link_libraries(faceRec ${OpenCV_LIBS})
