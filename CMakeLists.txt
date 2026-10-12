cmake_minimum_required(VERSION 3.16)
project(kernc VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release)
endif()

# The interpreters must do IEEE float math exactly as written: forbid the C++
# compiler from fusing a*b+c into FMA, otherwise the "oracle" itself would drift.
if(MSVC)
  add_compile_options(/W4 /EHsc /utf-8 /fp:precise /wd4100)
else()
  add_compile_options(-Wall -Wextra -ffp-contract=off -fno-fast-math)
endif()

add_library(kernclib STATIC
  src/lexer.cpp src/parser.cpp src/typecheck.cpp src/ir.cpp src/lower.cpp
  src/semantics.cpp src/passes.cpp src/ptx.cpp src/launch.cpp src/interp.cpp
  src/driver.cpp)
target_include_directories(kernclib PUBLIC src)

add_executable(kernc src/main.cpp)
target_link_libraries(kernc PRIVATE kernclib)

add_executable(kernc_unit_tests tests/unit/unit_tests.cpp)
target_link_libraries(kernc_unit_tests PRIVATE kernclib)

# Put executables in build/bin on every platform / generator
# (Visual Studio would otherwise add a Release/ or Debug/ sub-folder).
set_target_properties(kernc kernc_unit_tests PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/$<0:>")

enable_testing()
add_test(NAME unit_tests COMMAND kernc_unit_tests)
