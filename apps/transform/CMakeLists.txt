if (BUILD_APPLICATIONS)


set(trans_compile_defs "-DVERBOSE")
set(trans_compile_flags "")
set(trans_link_flags "")
set(trans_link_libs "m")



if (CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
	set(trans_compile_defs "${trans_compile_defs};-DHAVE_INTEL_ICC")
	if (PRINT_VEC_REPORT)
		set(trans_compile_flags "${trans_compile_flags};-qopt-report=5;-qopt-report-phase=vec;-qopt-report-file=stdout")
	endif(PRINT_VEC_REPORT)

elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
	if (PRINT_VEC_REPORT)
		set(trans_compile_flags "${trans_compile_flags};-ftree-vectorizer-verbose=4")
	endif(PRINT_VEC_REPORT)
endif()


set(TRANSFORM_APPS transform convert auc_pr_roc)

foreach( exec IN LISTS TRANSFORM_APPS)
	add_executable(${exec} ${exec}.cpp)
	foreach (cdef IN LISTS trans_compile_defs)
		target_compile_definitions(${exec} PRIVATE ${cdef})
	endforeach(cdef)
	foreach (cflgs IN LISTS trans_compile_flags)
		target_compile_options(${exec} PRIVATE ${cflgs})
	endforeach(cflgs)
	target_link_libraries(${exec} ${trans_link_flags} ${trans_link_libs} ${EXTRA_LIBS})

	if (USE_MPI)
		target_compile_definitions(${exec} PRIVATE -DUSE_MPI)
		# MPI_LIBRARIES already in EXTRA_LIBS
	endif(USE_MPI)
endforeach(exec)




endif(BUILD_APPLICATIONS)


