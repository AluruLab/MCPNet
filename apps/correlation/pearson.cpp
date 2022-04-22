/*
 * Copyright 2020 Georgia Tech Research Corporation
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
#include "splash/transform/zscore.hpp"
#include "splash/io/EXPMatrixWriter.hpp"
#include "splash/ds/aligned_matrix.hpp"
#include "splash/kernel/dotproduct.hpp"
#include "splash/patterns/pattern.hpp"

#include "splash/io/matrix_io.hpp"

#include "mcp/correlation/Pearson.hpp"

#ifdef USE_OPENMP
#include <omp.h>
#endif

class app_parameters : public parameters_base {
	public:

		app_parameters() {}
		virtual ~app_parameters() {}

		virtual void config(CLI::App& app) {}
		virtual void print(const char * prefix) {}
};



int main(int argc, char* argv[]) {

	//==============  PARSE INPUT =====================
	CLI::App app{"Pearson Correlation"};

	// handle MPI (TODO: replace with MXX later)
	splash::io::mpi_parameters mpi_params(argc, argv);
	mpi_params.config(app);

	// set up CLI parsers.
	splash::io::common_parameters common_params;
	app_parameters app_params;

	common_params.config(app);
	app_params.config(app);

	// parse
	CLI11_PARSE(app, argc, argv);

	// print out, for fun.
	FMT_ROOT_PRINT_RT("command line: ");
	for (int i = 0; i < argc; ++i) {
		FMT_ROOT_PRINT("{} ", argv[i]);
	}
	FMT_ROOT_PRINT("\n");


#ifdef USE_OPENMP
	// omp_set_dynamic(0);
	omp_set_num_threads(common_params.num_threads);
	FMT_PRINT_RT("omp num threads {}.  user threads {}\n", omp_get_max_threads(), common_params.num_threads);
#endif

	// =============== SETUP INPUT ===================
	// NOTE: input data is replicated on all MPI procs.
	splash::ds::aligned_matrix<double> input;
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
		input = read_matrix<double>(common_params.input, 
			"array",
			common_params.num_vectors, common_params.vector_size,
			genes, samples, common_params.skip);
	}
	etime = getSysTime();
	FMT_ROOT_PRINT("Load data in {} sec\n", get_duration_s(stime, etime));
	// input.print("INPUT: ");

	// ===== DEBUG ====== WRITE OUT INPUT =========
// {	// NOTE: rank 0 writes out.
// 	stime = getSysTime();
// 		// write to file.  MPI enabled.  Not thread enabled.
// 	write_matrix("input.exp", "array", genes, samples, input);
// 	etime = getSysTime();
// 	FMT_ROOT_PRINT("dump input in {} sec\n", get_duration_s(stime, etime));
// }

	if (mpi_params.rank == 0) {
		mpi_params.print("[PARAM] ");
		common_params.print("[PARAM] ");
		app_params.print("[PARAM] ");
	}

	// =============== NORMALIZE ==========================
	// every process normalize fully.
	// TODO: each rank does its own normalization then combine.

	// ------------ normalize -----------
	stime = getSysTime();
	// ---- create a VV2S kernel
	splash::pattern::Transform<splash::ds::aligned_matrix<double>, 
		splash::kernel::StandardScore<double, double, false>,
		splash::ds::aligned_matrix<double>> normalizer;
	splash::kernel::StandardScore<double, double, false> zscore;
	splash::ds::aligned_matrix<double> normalized(input.rows(), input.columns());
	normalizer(input, zscore, normalized);

	etime = getSysTime();
	FMT_ROOT_PRINT("Normalization Partitioned in {} sec\n", get_duration_s(stime, etime));
	//  normalized.print("NORM: ");

//	===== DEBUG ====== WRITE OUT normalized =========
// {	// NOTE: rank 0 writes out.
// 	stime = getSysTime();
// 		// write to file.  MPI enabled.  Not thread enabled.
// 	write_matrix("normal.exp", "array", genes, samples, normalized);
// 	etime = getSysTime();
// 	FMT_ROOT_PRINT("dump input in {} sec\n", get_duration_s(stime, etime));
// }

	// =============== PARTITION and RUN ===================
	stime = getSysTime();
	splash::ds::aligned_matrix<double> output(normalized.rows(), normalized.rows());
	using kernel_type = wave::correlation::PearsonKernel<double>;
	kernel_type correlation;
	splash::pattern::InnerProduct<splash::ds::aligned_matrix<double>, 
		kernel_type,
		splash::ds::aligned_matrix<double>> correlator;
	correlator(normalized, normalized, correlation, output);
	etime = getSysTime();
	FMT_ROOT_PRINT("Correlated in {} sec\n", get_duration_s(stime, etime));

	// =============== WRITE OUT RESULTS ==============
	// NOTE: rank 0 writes out.
	stime = getSysTime();
	write_matrix_distributed(common_params.output, "array", genes, genes, output);
	etime = getSysTime();
	FMT_ROOT_PRINT("Output in {} sec\n", get_duration_s(stime, etime));
	FMT_FLUSH();
	
	return 0;
}
