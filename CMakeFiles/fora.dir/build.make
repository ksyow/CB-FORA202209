# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
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
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/ksyow/Desktop/minimum-core

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/ksyow/Desktop/minimum-core

# Include any dependencies generated for this target.
include CMakeFiles/fora.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/fora.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/fora.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/fora.dir/flags.make

CMakeFiles/fora.dir/fora.cpp.o: CMakeFiles/fora.dir/flags.make
CMakeFiles/fora.dir/fora.cpp.o: fora.cpp
CMakeFiles/fora.dir/fora.cpp.o: CMakeFiles/fora.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/ksyow/Desktop/minimum-core/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/fora.dir/fora.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/fora.dir/fora.cpp.o -MF CMakeFiles/fora.dir/fora.cpp.o.d -o CMakeFiles/fora.dir/fora.cpp.o -c /home/ksyow/Desktop/minimum-core/fora.cpp

CMakeFiles/fora.dir/fora.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/fora.dir/fora.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/ksyow/Desktop/minimum-core/fora.cpp > CMakeFiles/fora.dir/fora.cpp.i

CMakeFiles/fora.dir/fora.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/fora.dir/fora.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/ksyow/Desktop/minimum-core/fora.cpp -o CMakeFiles/fora.dir/fora.cpp.s

CMakeFiles/fora.dir/mylib.cpp.o: CMakeFiles/fora.dir/flags.make
CMakeFiles/fora.dir/mylib.cpp.o: mylib.cpp
CMakeFiles/fora.dir/mylib.cpp.o: CMakeFiles/fora.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/ksyow/Desktop/minimum-core/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/fora.dir/mylib.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/fora.dir/mylib.cpp.o -MF CMakeFiles/fora.dir/mylib.cpp.o.d -o CMakeFiles/fora.dir/mylib.cpp.o -c /home/ksyow/Desktop/minimum-core/mylib.cpp

CMakeFiles/fora.dir/mylib.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/fora.dir/mylib.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/ksyow/Desktop/minimum-core/mylib.cpp > CMakeFiles/fora.dir/mylib.cpp.i

CMakeFiles/fora.dir/mylib.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/fora.dir/mylib.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/ksyow/Desktop/minimum-core/mylib.cpp -o CMakeFiles/fora.dir/mylib.cpp.s

CMakeFiles/fora.dir/config.cpp.o: CMakeFiles/fora.dir/flags.make
CMakeFiles/fora.dir/config.cpp.o: config.cpp
CMakeFiles/fora.dir/config.cpp.o: CMakeFiles/fora.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/ksyow/Desktop/minimum-core/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/fora.dir/config.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/fora.dir/config.cpp.o -MF CMakeFiles/fora.dir/config.cpp.o.d -o CMakeFiles/fora.dir/config.cpp.o -c /home/ksyow/Desktop/minimum-core/config.cpp

CMakeFiles/fora.dir/config.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/fora.dir/config.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/ksyow/Desktop/minimum-core/config.cpp > CMakeFiles/fora.dir/config.cpp.i

CMakeFiles/fora.dir/config.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/fora.dir/config.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/ksyow/Desktop/minimum-core/config.cpp -o CMakeFiles/fora.dir/config.cpp.s

# Object files for target fora
fora_OBJECTS = \
"CMakeFiles/fora.dir/fora.cpp.o" \
"CMakeFiles/fora.dir/mylib.cpp.o" \
"CMakeFiles/fora.dir/config.cpp.o"

# External object files for target fora
fora_EXTERNAL_OBJECTS =

fora: CMakeFiles/fora.dir/fora.cpp.o
fora: CMakeFiles/fora.dir/mylib.cpp.o
fora: CMakeFiles/fora.dir/config.cpp.o
fora: CMakeFiles/fora.dir/build.make
fora: CMakeFiles/fora.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/ksyow/Desktop/minimum-core/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Linking CXX executable fora"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/fora.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/fora.dir/build: fora
.PHONY : CMakeFiles/fora.dir/build

CMakeFiles/fora.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/fora.dir/cmake_clean.cmake
.PHONY : CMakeFiles/fora.dir/clean

CMakeFiles/fora.dir/depend:
	cd /home/ksyow/Desktop/minimum-core && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/ksyow/Desktop/minimum-core /home/ksyow/Desktop/minimum-core /home/ksyow/Desktop/minimum-core /home/ksyow/Desktop/minimum-core /home/ksyow/Desktop/minimum-core/CMakeFiles/fora.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/fora.dir/depend

