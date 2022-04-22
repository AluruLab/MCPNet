/*
 * Copyright 2021 Georgia Tech Research Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * 
 * Author(s): Tony C. Pan
 */

#include <sstream>
#include <string>

#include "CLI/CLI.hpp"
#include "splash/io/CLIParserCommon.hpp"
#include "splash/io/parameters_base.hpp"
#include "splash/utils/benchmark.hpp"
#include "splash/utils/report.hpp"
#include "splash/ds/aligned_matrix.hpp"

#include "mcp/filter/stencil.hpp"

#include "splash/io/matrix_io.hpp"


#ifdef USE_OPENMP
#include <omp.h>
#endif



int main(int argc, char* argv[]) {

	//==============  PARSE INPUT =====================
	CLI::App app{"Convert File Formats.  supports EXP, CSV, HDF5, and randomly generated data"};

	// handle MPI (TODO: replace with MXX later)
	splash::io::mpi_parameters mpi_params(argc, argv);
	mpi_params.config(app);
	if (mpi_params.procs > 1) {
		FMT_PRINT_ERR("ERROR: convert should not be used with MPI.\n");
		return -2;
	}

	// set up CLI parsers.
	splash::io::common_parameters common_params;

	common_params.config(app);

	// parse
	CLI11_PARSE(app, argc, argv);

	// print out, for fun.
	FMT_ROOT_PRINT_RT("command line: ");
	for (int i = 0; i < argc; ++i) {
		FMT_ROOT_PRINT("{} ", argv[i]);
	}
	FMT_ROOT_PRINT("\n");


#ifdef USE_OPENMP
	omp_set_num_threads(common_params.num_threads);
#endif

	// =============== SETUP INPUT ===================
	// NOTE: input data is replicated on all MPI procs.
	using MatrixType = splash::ds::aligned_matrix<double>;
	MatrixType input;
	std::vector<std::string> genes;
	std::vector<std::string> samples;
	
	auto stime = getSysTime();
	auto etime = getSysTime();

	stime = getSysTime();
	if (common_params.random) {
		input = make_random_matrix(common_params.rseed, 
			common_params.rmin, common_params.rmax, 
			common_params.num_vectors, common_params.vector_size,
			genes, samples);
	} else {
		input = read_matrix<double>(common_params.input, "array", 
			common_params.num_vectors, common_params.vector_size,
			genes, samples);
	}
	etime = getSysTime();
	FMT_ROOT_PRINT("Load data from {} in {} sec\n", common_params.input, get_duration_s(stime, etime));

	if (mpi_params.rank == 0) {
		mpi_params.print("[PARAM] ");
		common_params.print("[PARAM] ");
	}


	// ===== DEBUG ====== WRITE OUT INPUT =========
	{	// NOTE: rank 0 writes out.
		stime = getSysTime();
		// write to file.  MPI enabled.  Not thread enabled.
		if (mpi_params.rank == 0)
			write_matrix(common_params.output, "array", genes, samples, input);
		etime = getSysTime();
		FMT_ROOT_PRINT("converted to {} in {} sec\n", common_params.output, get_duration_s(stime, etime));
	}

	FMT_FLUSH();


	return 0;
}
