cmake_minimum_required(VERSION 3.5)
project(point_pillars)
add_subdirectory(pybind11)
pybind11_add_module(point_pillars SHARED src/point_pillars.cpp)