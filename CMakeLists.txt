# CMakeLists.txt template

cmake_minimum_required(VERSION 2.8)


# project settings
project(mcpnet)

set (mcpnet_VERSION_MAJOR 1)
set (mcpnet_VERSION_MINOR 0)
set (mcpnet_VERSION_PATCH 0)
set (mcpnet_VERSION_TWEAK 0)

INCLUDE(CMakeDependentOption)

##### General Compilation Settings

### REQUIRE C++11
# Initialize CXXFLAGS.
set(BASE_OPT_FLAGS "-ffast-math -funroll-loops")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wuninitialized --std=c++11 -fPIC")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${BASE_OPT_FLAGS}")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} ${BASE_OPT_FLAGS}")

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wuninitialized --std=c11 -fPIC")
set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -O0")
set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} ${BASE_OPT_FLAGS}")
set(CMAKE_C_FLAGS_RELWITHDEBINFO "${CMAKE_C_FLAGS_RELWITHDEBINFO} ${BASE_OPT_FLAGS}")


set(EXT_PROJECTS_DIR ${PROJECT_SOURCE_DIR}/ext)
set(CMAKE_MODULE_DIR ${PROJECT_SOURCE_DIR}/ext/cmake-utils)
#set(CMAKE_MODULE_PATH ${CMAKE_MODULE_DIR} ${CMAKE_MODULE_PATH})

# Compiler-specific C++11 activation.
if (CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
# below 2 lines are only for cmake < 2.8.10
   execute_process(
     COMMAND ${CMAKE_CXX_COMPILER} -dumpversion OUTPUT_VARIABLE BLISS_CXX_COMPILER_VERSION)
  if(BLISS_CXX_COMPILER_VERSION VERSION_LESS "16.0")
    message(FATAL_ERROR "Intel (icpc) compiler version 16.0 or later is required.")
  endif()

  #add_definitions(-Wabi)

  SET(SUPPORTS_SANITIZER OFF CACHE INTERNAL "compiler supports sanitizer?")
  SET(SUPPORTS_COVERAGE OFF CACHE INTERNAL "compiler supports coverage?")

# intel dependency on gcc.  not compatible with gcc 5.  should search.
#  set(GCC_BINARY "gcc" CACHE PATH "Path to gcc binary. Requires version 4.9 or earlier.  Intel compiler uses GCC headers.")
#  EXECUTE_PROCESS( COMMAND ${GCC_BINARY} --version OUTPUT_VARIABLE gcc_full_version_string )
#  string (REGEX REPLACE ".* ([0-9]+\\.[0-9]+\\.[0-9]+).*" "\\1" GCC_VERSION ${gcc_full_version_string})
  
#  if (GCC_VERSION VERSION_GREATER 5.0.0 OR GCC_VERSION VERSION_EQUAL 5.0.0)
#    message(FATAL_ERROR "${PROJECT_NAME} is using Intel compiler, which requires gcc headers from version 4.9 or earlier.")
#  endif ()
#  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -gcc-name=${GCC_BINARY}")
  OPTION(PRINT_VEC_REPORT "Intel compiler vector report" ON)

elseif (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
# next 2 lines needed for cmake < 2.8.10
   EXECUTE_PROCESS( COMMAND ${CMAKE_CXX_COMPILER} --version OUTPUT_VARIABLE
   gcc_full_version_string )
   string (REGEX REPLACE ".* ([0-9]+\\.[0-9]+\\.[0-9]+).*" "\\1"
    BLISS_CXX_COMPILER_VERSION
    ${gcc_full_version_string})

  if(BLISS_CXX_COMPILER_VERSION VERSION_LESS "4.8.1")
    message(FATAL_ERROR "GNU CXX compiler version 4.8.1 or later is required.
      Found ${CMAKE_CXX_COMPILER_VERSION}")
  endif()

  SET(SUPPORTS_SANITIZER ON CACHE INTERNAL "compiler supports sanitizer?")
  SET(SUPPORTS_COVERAGE ON CACHE INTERNAL "compiler supports coverage?")
  
  # gcc 4.9 colorize via: -fdiagnostics-color=always
  
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  # these 3 lines are needed because cmake < 2.8.10
  EXECUTE_PROCESS( COMMAND ${CMAKE_CXX_COMPILER} --version OUTPUT_VARIABLE
     clang_full_version_string )
  string (REGEX REPLACE ".*clang version ([0-9]+\\.[0-9]+).*" "\\1"
   BLISS_CXX_COMPILER_VERSION ${clang_full_version_string})

  if(BLISS_CXX_COMPILER_VERSION VERSION_LESS "3.5")
    message(FATAL_ERROR "Clang++ compiler version 3.5 or later is required.")
  endif()
    
  #add_definitions(-Wabi)
 
  # clang dependency on gcc?  not compatible with gcc 5.  should search.
#  set(GCC_BINARY "/usr/bin/gcc" CACHE PATH "Path to gcc binary. Requires version 4.9 or earlier due to ABI change.  Clang uses GCC headers.")
#  EXECUTE_PROCESS( COMMAND ${GCC_BINARY} --version OUTPUT_VARIABLE gcc_full_version_string )
#  string (REGEX REPLACE ".* ([0-9]+\\.[0-9]+\\.[0-9]+).*" "\\1" GCC_VERSION ${gcc_full_version_string})
#  
#  if (GCC_VERSION VERSION_GREATER 5.0.0 OR GCC_VERSION VERSION_EQUAL 5.0.0)
#    message(FATAL_ERROR "${PROJECT_NAME} is using Clang compiler, which requires gcc headers from version 4.9 or earlier.")
#  endif ()
    
    # message(FATAL_ERROR "${PROJECT_NAME} currently does not support clang due to include path issues")
#  set(_CMAKE_TOOLCHAIN_PREFIX "llvm-")
  
  get_filename_component(CLANG_COMPILER_DIR ${CMAKE_CXX_COMPILER} PATH)
  # not needed
  # include_directories("${CLANG_COMPILER_DIR}/../lib/clang/${CMAKE_CXX_COMPILER_VERSION}/include")		      # this is here for SSE headers.
  # not needed  
  # include_directories("${CLANG_COMPILER_DIR}/../include")  # this is here for omp header, if one is installed.
  link_directories(${CLANG_COMPILER_DIR}/../lib)
  
  set(CMAKE_EXE_LINK_FLAGS "${CMAKE_EXE_LINK_FLAGS} -lstdc++")
  set(CMAKE_MODULE_LINK_FLAGS "${CMAKE_MODULE_LINK_FLAGS} -lstdc++")
  set(CMAKE_SHARED_LINK_FLAGS "${CMAKE_SHARED_LINK_FLAGS} -lstdc++")
  set(CMAKE_STATIC_LINK_FLAGS "${CMAKE_STATIC_LINK_FLAGS} -lstdc++")
    
  # clang uses gcc headers, and can find most by itself.
  SET(SUPPORTS_SANITIZER ON CACHE INTERNAL "compiler supports sanitizer?")
  SET(SUPPORTS_COVERAGE ON CACHE INTERNAL "compiler supports coverage?")

else ()
  message(FATAL_ERROR "Your C++ compiler is not supported.")
endif ()
### from http://stackoverflow.com/questions/10984442/how-to-detect-c11-support-of-a-compiler-with-cmake


IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING
      "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel."
      FORCE)
ENDIF(NOT CMAKE_BUILD_TYPE)

# Add these standard paths to the search paths for FIND_LIBRARY
# to find libraries from these locations first
if(UNIX)
  set(CMAKE_LIBRARY_PATH "${CMAKE_LIBRARY_PATH} /lib /usr/lib")
endif()

# --------------------------------------------------------------
# Indicate CMake 2.7 and above that we don't want to mix relative
#  and absolute paths in linker lib lists.
# Run "cmake --help-policy CMP0003" for more information.
# --------------------------------------------------------------
if(COMMAND cmake_policy)
  cmake_policy(SET CMP0003 NEW)

  #for recognizing numbers and boolean constants in if()
  if(CMAKE_VERSION VERSION_GREATER 2.6.4)
    cmake_policy(SET CMP0012 NEW)
  endif()
endif()

# ----------------------------------------------------------------------------
# Build static or dynamic libs?
# Default: dynamic libraries
# ----------------------------------------------------------------------------
OPTION(BUILD_SHARED_LIBS "Build shared libraries (.dll/.so) instead of static ones (.lib/.a)" ON)


include("${CMAKE_MODULE_DIR}/ExtraWarnings.cmake")
include("${CMAKE_MODULE_DIR}/GCCFilter.cmake")
include("${CMAKE_MODULE_DIR}/CompilerProfiling.cmake")
include("${CMAKE_MODULE_DIR}/OpenMP.cmake")
include("${CMAKE_MODULE_DIR}/MPI.cmake")
include("${CMAKE_MODULE_DIR}/SIMD.cmake")
include("${CMAKE_MODULE_DIR}/MKL.cmake")
include("${CMAKE_MODULE_DIR}/HDF5.cmake")
include("${CMAKE_MODULE_DIR}/CodeAnalysis.cmake")
include("${CMAKE_MODULE_DIR}/Sanitizer.cmake")


###### EXECUTABLE CATEGORIES
#  organization:  3 levels:  benchmark (coarse grain), profile (fine grain), debug (include coverage and sanitizer).  profile and debug are mutually exclusive.
### executable option organization:
# build_applications - useful applications
# build_tests -  both can be on.
# BUILD_BENCHMARKS - enables benchmark reporting at application level

# Check if the user want to build sample applications
OPTION(BUILD_APPLICATIONS "whether applications should be built" ON)
OPTION(BUILD_TESTS "whether tests should be built" OFF)
OPTION(BUILD_BENCHMARKS "whether benchmarks should be built" OFF)

# library usage.
OPTION(USE_ZLIB "whether mcpnet should use ZLIB" OFF)

###### Executable and Libraries
# Save libs and executables in the same place
set(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR}/lib CACHE PATH "Output directory for libraries" )
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin CACHE PATH "Output directory for applications" )


# add the binary tree to the search path for include files
# so that we will find TutorialConfig.h
include_directories("${PROJECT_SOURCE_DIR}/include")
#include_directories("${CMAKE_BINARY_DIR}")
include_directories("${EXT_PROJECTS_DIR}/splash/include") 
include_directories("${EXT_PROJECTS_DIR}/splash/ext/fmt/include") 
include_directories("${EXT_PROJECTS_DIR}/CLI11/include") 


if (BUILD_APPLICATIONS)
  add_subdirectory(apps)
endif()

if (BUILD_TESTS)
  add_subdirectory(test)
endif()

#if (BUILD_BENCHMARKS)
#  add_subdirectory(benchmarks)
#endif()


# Install Settings


# Installer Settings


# Test Settings
