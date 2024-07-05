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

template<typename DataType>
class app_parameters : public parameters_base {
	public:
		DataType target_val;  // target_value_to_replace

		app_parameters(DataType const & target = 0) : 
                target_val(target) {}
        app_parameters(const app_parameters<double>& other){
                target_val = DataType(other.target_val);
        }
		virtual ~app_parameters() {}

		virtual void config(CLI::App& app) {
			app.add_option("--target-value", target_val, "new value to use.")->group("filter");
		}
		virtual void print(const char * prefix) const {
            FMT_ROOT_PRINT("{} new value       : {}\n", prefix, target_val);
		}
        
};

template<typename DataType>
void run(splash::io::common_parameters& common_params,
         splash::io::mpi_parameters& mpi_params,
         app_parameters<DataType>& app_params ){

	// =============== SETUP INPUT ===================
	// NOTE: input data is replicated on all MPI procs.
	using MatrixType = splash::ds::aligned_matrix<DataType>;
	MatrixType input, pv;
	std::vector<std::string> genes;
	std::vector<std::string> samples;
	
	auto stime = getSysTime();
	auto etime = getSysTime();

	stime = getSysTime();
	if (common_params.random) {
		input = make_random_matrix<DataType>(common_params.rseed, 
			common_params.rmin, common_params.rmax, 
			common_params.num_vectors, common_params.vector_size,
			genes, samples);
	} else {
		input = read_matrix<DataType>(common_params.input, common_params.h5_group,
			common_params.num_vectors, common_params.vector_size,
			genes, samples, common_params.skip, 1, common_params.h5_gene_key,
            common_params.h5_samples_key, common_params.h5_matrix_key);
	}
	etime = getSysTime();
	FMT_ROOT_PRINT("Load data in {} sec\n", get_duration_s(stime, etime));


	// ===== DEBUG ====== WRITE OUT INPUT =========
// {	// NOTE: rank 0 writes out.
// 	stime = getSysTime();
// 		// write to file.  MPI enabled.  Not thread enabled.
// 	write_matrix("input2.exp", "array", genes, samples, input);
// 	etime = getSysTime();
// 	FMT_ROOT_PRINT("dump input in {} sec\n", get_duration_s(stime, etime));
// }
	// input.print("INPUT: ");

	if (mpi_params.rank == 0) {
		mpi_params.print("[PARAM] ");
		common_params.print("[PARAM] ");
		app_params.print("[PARAM] ");
	}

	// =============== PARTITION and RUN ===================
	stime = getSysTime();
	MatrixType output(input.rows(), input.columns());


    // correlation close to 0 is bad.
	// input is full matrix, output is distributed.
	using ThresholdType = ::mcp::stencil::Diagonal<MatrixType>;
	ThresholdType thresholder(app_params.target_val);
	thresholder(input, output);


	etime = getSysTime();
	FMT_ROOT_PRINT("Stenciled in {} sec\n", get_duration_s(stime, etime));
	

	// =============== WRITE OUT ==============
	// NOTE: rank 0 writes out.
	stime = getSysTime();
	write_matrix_distributed(common_params.output, "array", genes, samples, output);
	etime = getSysTime();
	FMT_ROOT_PRINT("Output in {} sec\n", get_duration_s(stime, etime));
	FMT_FLUSH();



}


int main(int argc, char* argv[]) {

	//==============  PARSE INPUT =====================
	CLI::App app{"Stenciling Diagonal"};

	// handle MPI (TODO: replace with MXX later)
	splash::io::mpi_parameters mpi_params(argc, argv);
	mpi_params.config(app);

	// set up CLI parsers.
	splash::io::common_parameters common_params;
	app_parameters<double> app_params;

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
	omp_set_num_threads(common_params.num_threads);
#endif

    if(common_params.use_single) {
        app_parameters<float> flapp_params(app_params);
        run<float>(common_params, mpi_params, flapp_params);
    } else {
        run<double>(common_params, mpi_params, app_params);
    }

	return 0;
}
