cmake_minimum_required(VERSION 3.16)

project(inverse_kinematics VERSION 1.0 DESCRIPTION "MuJoCo project with planner")

# Define MuJoCo path
set(MUJOCO_PATH "/usr/local/mujoco") 

# Add the path to MuJoCo headers
include_directories(${MUJOCO_PATH}/include)

# Add the path to GLFW library
set(GLFW_LIB_PATH "/home/mateus/mujoco/build/lib")  
link_directories(${GLFW_LIB_PATH})

find_package(GLEW REQUIRED)
find_package(Eigen3 REQUIRED)

add_executable(inverse_kinematics basic.cc)  

target_link_libraries(inverse_kinematics
    ${MUJOCO_PATH}/lib/libmujoco.so 
    GLEW::GLEW
    glfw
    ${GLFW_LIB_PATH}/libglfw3.a  # Link GLFW explicitly
)
