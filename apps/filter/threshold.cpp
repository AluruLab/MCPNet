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
#include "splash/ds/aligned_matrix.hpp"
#include "splash/patterns/pattern.hpp"

#include "mcp/filter/threshold.hpp"

#include "splash/io/matrix_io.hpp"


#ifdef USE_OPENMP
#include <omp.h>
#endif

template<typename DataType>
class app_parameters : public parameters_base {
	public:
		DataType lower_thresh;  // default value depends on usage.
		DataType upper_thresh;  
        // DataType pv_thresh;   // <= pv_thresh
        std::string pv_input;
		bool invert;

		app_parameters(DataType const & min = std::numeric_limits<DataType>::lowest(), 
            DataType const & max = std::numeric_limits<DataType>::max()) : 
                lower_thresh(min), upper_thresh(max), invert(false) {}
        app_parameters(const app_parameters<double>& other){
            lower_thresh = DataType(other.lower_thresh);
            upper_thresh = DataType(other.upper_thresh);
            pv_input = other.pv_input;
            invert = other.invert;
        }

		virtual ~app_parameters() {}

		virtual void config(CLI::App& app) {
			auto lower_thresh_opt = app.add_option("--lower", lower_thresh, "minimum value threshold.  for 2 tailed only")->group("threshold");
			auto upper_thresh_opt = app.add_option("--upper", upper_thresh, "maximum value threshold.")->group("threshold");
			// auto pv_thresh_opt = app.add_option("--max-pv", pv_thresh, "P-value threshold.")->group("threshold");
			app.add_option("-p,--p-value-input", pv_input, "P-value filename.")->group("threshold");
            app.add_flag("--invert", invert, "invert threshold range (select extrema. Default false.) ");
			// pv_thresh_opt->needs(pv_input_opt);
            lower_thresh_opt->needs(upper_thresh_opt);
		}
		virtual void print(const char * prefix) const {
            FMT_ROOT_PRINT("{} value low threshold  : {}\n", prefix, lower_thresh); 
            FMT_ROOT_PRINT("{} value hi threshold   : {}\n", prefix, upper_thresh); 
            // FMT_ROOT_PRINT("{} pvalue threshold      : {}\n", prefix, pv_thresh); 
            FMT_ROOT_PRINT("{} pvalue input         : {}\n", prefix, pv_input.c_str()); 
			FMT_ROOT_PRINT("{} invert selection     : {}\n", prefix, (invert ? "Y" : "N"));
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
		input = read_matrix<DataType>(common_params.input, 
            common_params.h5_group, common_params.num_vectors, 
            common_params.vector_size, genes, samples, common_params.skip, 1,
            common_params.h5_gene_key, common_params.h5_samples_key,
            common_params.h5_matrix_key);
	}
	etime = getSysTime();
	FMT_ROOT_PRINT("Load data in {} sec\n", get_duration_s(stime, etime));

    // load p value if there is one.
    if (app_params.pv_input.length() > 0) {
        stime = getSysTime();
        std::vector<std::string> genes2;
	    std::vector<std::string> samples2;
        pv = read_matrix<DataType>(app_params.pv_input, 
            "/", 
			common_params.num_vectors, common_params.vector_size,
			genes2, samples2, common_params.skip, 1, common_params.h5_gene_key,
            common_params.h5_samples_key, common_params.h5_matrix_key);
			// "array",
            // common_params.num_vectors, common_params.vector_size,
            // genes2, samples2);
        etime = getSysTime();
        FMT_ROOT_PRINT("Load pvalue data in {} sec\n", get_duration_s(stime, etime));
    }

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

    bool pv_set = app_params.pv_input.length() > 0;

	size_t removed;
    // correlation close to 0 is bad.
	if (app_params.invert) {
		if (pv_set) { // using pvalue threshold
			using KernelType = mcp::kernel::inverted_threshold2<DataType, DataType>;
			KernelType kernel(app_params.lower_thresh, app_params.upper_thresh, 0.0);
			using ThresholdType = ::splash::pattern::GlobalBinaryOp<MatrixType, MatrixType, KernelType, MatrixType>;
			ThresholdType thresholder;
			thresholder(input, pv, kernel, output);
			removed = thresholder.processed;
		} else {
			using KernelType = mcp::kernel::inverted_threshold<DataType>;
			KernelType kernel(app_params.lower_thresh, app_params.upper_thresh, 0.0);
			using ThresholdType = ::splash::pattern::GlobalTransform<MatrixType, KernelType, MatrixType>;
			ThresholdType thresholder;
			thresholder(input, kernel, output);
			removed = thresholder.processed;
		}
	} else {
		if (pv_set) { // using pvalue threshold
			using KernelType = mcp::kernel::threshold2<DataType, DataType>;
			KernelType kernel(app_params.lower_thresh, app_params.upper_thresh, 0.0);
			using ThresholdType = ::splash::pattern::GlobalBinaryOp<MatrixType, MatrixType, KernelType, MatrixType>;
			ThresholdType thresholder;
			thresholder(input, pv, kernel, output);
			removed = thresholder.processed;
		} else {
			using KernelType = mcp::kernel::threshold<DataType>;
			KernelType kernel(app_params.lower_thresh, app_params.upper_thresh, 0.0);
			using ThresholdType = ::splash::pattern::GlobalTransform<MatrixType, KernelType, MatrixType>;
			ThresholdType thresholder;
			thresholder(input, kernel, output);
			removed = thresholder.processed;
		}
		/* code */
	}
	

	etime = getSysTime();
	FMT_ROOT_PRINT("Thresholded in {} sec\n", get_duration_s(stime, etime));
	FMT_ROOT_PRINT("Threshold removed {} elements\n", removed);
	

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
	CLI::App app{"Thresholding"};

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
