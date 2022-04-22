if (BUILD_APPLICATIONS)


set(filter_compile_defs "-DVERBOSE")
set(filter_compile_flags "")
set(filter_link_flags "")
set(filter_link_libs "m")


#if (USE_ZLIB)
#	set(filter_compile_defs "${filter_compile_defs};-DCOMPRESSED_INPUT")
#	set(filter_link_libs "z;${filter_link_libs}")
#endif(USE_ZLIB)

#if (USE_MKL)
#	set(filter_compile_defs "${filter_compile_defs};-DWITH_MKL;${MKL_DEFINES}")
#	set(filter_compile_flags "${MKL_COMPILE_FLAGS}")
#	set(filter_link_flags "${MKL_LINK_FLAGS}")
#	include_directories(${MKL_INCLUDE_DIRS})
#	link_directories(${MKL_LIB_DIRS})
#	set(filter_link_libs "${MKL_LIBS};${filter_link_libs}")
#endif(USE_MKL)

if (CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
	set(filter_compile_defs "${filter_compile_defs};-DHAVE_INTEL_ICC")
	if (PRINT_VEC_REPORT)
		set(filter_compile_flags "${filter_compile_flags};-qopt-report=5;-qopt-report-phase=vec;-qopt-report-file=stdout")
	endif(PRINT_VEC_REPORT)

elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
	if (PRINT_VEC_REPORT)
		set(filter_compile_flags "${filter_compile_flags};-ftree-vectorizer-verbose=4")
	endif(PRINT_VEC_REPORT)
endif()


set(CORRELATION_APPS threshold dpi diagonal select mcp)

foreach( exec IN LISTS CORRELATION_APPS)
	add_executable(${exec} ${exec}.cpp)
	foreach (cdef IN LISTS filter_compile_defs)
		target_compile_definitions(${exec} PRIVATE ${cdef})
	endforeach(cdef)
	foreach (cflgs IN LISTS filter_compile_flags)
		target_compile_options(${exec} PRIVATE ${cflgs})
	endforeach(cflgs)
	target_link_libraries(${exec} ${filter_link_flags} ${filter_link_libs} ${EXTRA_LIBS})

	if (USE_MPI)
		target_compile_definitions(${exec} PRIVATE -DUSE_MPI)
		# MPI_LIBRARIES already in EXTRA_LIBS
	endif(USE_MPI)
endforeach(exec)




endif(BUILD_APPLICATIONS)


