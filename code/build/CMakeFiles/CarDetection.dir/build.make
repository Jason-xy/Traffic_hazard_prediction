# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.10

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/jason/github/Traffic_hazard_prediction/code

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/jason/github/Traffic_hazard_prediction/code/build

# Include any dependencies generated for this target.
include CMakeFiles/CarDetection.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/CarDetection.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/CarDetection.dir/flags.make

CMakeFiles/CarDetection.dir/carDetection.cpp.o: CMakeFiles/CarDetection.dir/flags.make
CMakeFiles/CarDetection.dir/carDetection.cpp.o: ../carDetection.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/jason/github/Traffic_hazard_prediction/code/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/CarDetection.dir/carDetection.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/CarDetection.dir/carDetection.cpp.o -c /home/jason/github/Traffic_hazard_prediction/code/carDetection.cpp

CMakeFiles/CarDetection.dir/carDetection.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/CarDetection.dir/carDetection.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/jason/github/Traffic_hazard_prediction/code/carDetection.cpp > CMakeFiles/CarDetection.dir/carDetection.cpp.i

CMakeFiles/CarDetection.dir/carDetection.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/CarDetection.dir/carDetection.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/jason/github/Traffic_hazard_prediction/code/carDetection.cpp -o CMakeFiles/CarDetection.dir/carDetection.cpp.s

CMakeFiles/CarDetection.dir/carDetection.cpp.o.requires:

.PHONY : CMakeFiles/CarDetection.dir/carDetection.cpp.o.requires

CMakeFiles/CarDetection.dir/carDetection.cpp.o.provides: CMakeFiles/CarDetection.dir/carDetection.cpp.o.requires
	$(MAKE) -f CMakeFiles/CarDetection.dir/build.make CMakeFiles/CarDetection.dir/carDetection.cpp.o.provides.build
.PHONY : CMakeFiles/CarDetection.dir/carDetection.cpp.o.provides

CMakeFiles/CarDetection.dir/carDetection.cpp.o.provides.build: CMakeFiles/CarDetection.dir/carDetection.cpp.o


# Object files for target CarDetection
CarDetection_OBJECTS = \
"CMakeFiles/CarDetection.dir/carDetection.cpp.o"

# External object files for target CarDetection
CarDetection_EXTERNAL_OBJECTS =

CarDetection: CMakeFiles/CarDetection.dir/carDetection.cpp.o
CarDetection: CMakeFiles/CarDetection.dir/build.make
CarDetection: /usr/local/cuda-10.2/lib64/libcudart_static.a
CarDetection: /usr/lib/aarch64-linux-gnu/librt.so
CarDetection: /usr/local/lib/libjetson-inference.so
CarDetection: /usr/local/lib/libjetson-utils.so
CarDetection: /usr/local/cuda-10.2/lib64/libcudart_static.a
CarDetection: /usr/lib/aarch64-linux-gnu/librt.so
CarDetection: /usr/local/cuda-10.2/lib64/libnppicc.so
CarDetection: CMakeFiles/CarDetection.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/jason/github/Traffic_hazard_prediction/code/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable CarDetection"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/CarDetection.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/CarDetection.dir/build: CarDetection

.PHONY : CMakeFiles/CarDetection.dir/build

CMakeFiles/CarDetection.dir/requires: CMakeFiles/CarDetection.dir/carDetection.cpp.o.requires

.PHONY : CMakeFiles/CarDetection.dir/requires

CMakeFiles/CarDetection.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/CarDetection.dir/cmake_clean.cmake
.PHONY : CMakeFiles/CarDetection.dir/clean

CMakeFiles/CarDetection.dir/depend:
	cd /home/jason/github/Traffic_hazard_prediction/code/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/jason/github/Traffic_hazard_prediction/code /home/jason/github/Traffic_hazard_prediction/code /home/jason/github/Traffic_hazard_prediction/code/build /home/jason/github/Traffic_hazard_prediction/code/build /home/jason/github/Traffic_hazard_prediction/code/build/CMakeFiles/CarDetection.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/CarDetection.dir/depend
