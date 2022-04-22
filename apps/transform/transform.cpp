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
#include <unordered_set>

#include "CLI/CLI.hpp"
#include "splash/io/CLIParserCommon.hpp"
#include "splash/io/parameters_base.hpp"
#include "splash/utils/benchmark.hpp"
#include "splash/utils/report.hpp"
#include "splash/ds/aligned_matrix.hpp"
#include "splash/patterns/pattern.hpp"
#include "mcp/transform/clr.hpp"
#include "mcp/transform/stouffer.hpp"
#include "splash/transform/zscore.hpp"


#include "splash/io/matrix_io.hpp"


#ifdef USE_OPENMP
#include <omp.h>
#endif

class app_parameters : public parameters_base {
	public:
		enum method_type : int { CLR = 1, STOUFFER = 2 };

		method_type method;
		std::string tf_input;

		app_parameters() : method(STOUFFER) {}
		virtual ~app_parameters() {}

		virtual void config(CLI::App& app) {
            app.add_option("-m,--method", method, "Algorithm: CLR=1, Stouffer=2");
			app.add_option("-f,--tf-input", tf_input, "transcription factor file, 1 per line.");

		}
		virtual void print(const char * prefix) {
			FMT_ROOT_PRINT("{} TF input: {}\n", prefix, tf_input.c_str()); 
            FMT_ROOT_PRINT("{} computational method: {}\n", prefix, (method == CLR ? "CLR" :
				"Stouffer")); 
		}
};



int main(int argc, char* argv[]) {

	//==============  PARSE INPUT =====================
	CLI::App app{"Correlation Transform"};

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
		input = read_matrix<double>(common_params.input, 
			"array",
			common_params.num_vectors, common_params.vector_size,
			genes, samples);
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

	std::unordered_set<std::string> TFs;
	std::vector<bool> tfs;
	std::vector<std::string> tf_names;
	if (app_params.tf_input.length() > 0) {
		stime = getSysTime();

		// read the file and put into a set.
		std::ifstream ifs(app_params.tf_input);
		std::string line;
		while (std::getline(ifs, line)) {
			TFs.insert(splash::utils::trim(line));
		}
		ifs.close();
	
		// now check for each gene whether it's in TF list.
		for (size_t i = 0; i < genes.size(); ++i) {
			tfs.push_back(TFs.count(splash::utils::trim(genes[i])) > 0); // if this is one of the target TF, then safe it
			if (tfs.back() == true) {
				tf_names.push_back(genes[i]);
			}
		}

		etime = getSysTime();
		FMT_ROOT_PRINT("Load TF data in {} sec\n", get_duration_s(stime, etime));
		if (mpi_params.rank == 0) {
			FMT_ROOT_PRINT("Selected TFs:");
			for (auto TF : TFs) {
				FMT_ROOT_PRINT("\t{}\n", TF.c_str());
			}
		}
	}

	MatrixType tfs_to_genes(1, input.columns());    // tfs as rows
	MatrixType genes_to_tfs(input.rows(), 1);		// tfs as columns
	if (app_params.tf_input.length() > 0) {
		
		// select rows.
		tfs_to_genes.resize(tf_names.size(), input.columns());
		for (size_t i = 0, j = 0; i < genes.size(); ++i) {
			if (tfs[i]) {
				memcpy(tfs_to_genes.data(j), input.data(i), input.column_bytes());
				++j;
			}
		}
		genes_to_tfs = tfs_to_genes.local_transpose();
	}


	// =============== PARTITION and RUN ===================
	MatrixType output(1, input.columns());
	if (app_params.tf_input.length() > 0) {
		stime = getSysTime();
		splash::pattern::Transform<MatrixType, 
			splash::kernel::StandardScore<double, double, false>,
			MatrixType> normalizer;
		splash::kernel::StandardScore<double, double, false> zscore;

		// these are distributed so need to gather.
		MatrixType tf_norms;
		{
			MatrixType tf_normsD(tfs_to_genes.rows(), tfs_to_genes.columns());
			normalizer(tfs_to_genes, zscore, tf_normsD);
			tf_norms = tf_normsD.allgather();    // not efficient
		}
		MatrixType gene_norms;
		{
			MatrixType gene_normsDT(genes_to_tfs.rows(), genes_to_tfs.columns());
			normalizer(genes_to_tfs, zscore, gene_normsDT);

			// transposed to match tf_norms:  allgather, transpose
			// TODO: make this faster in the aligned_matrix code.
			gene_norms = gene_normsDT.allgather().local_transpose();  // not efficient.
		}

		etime = getSysTime();
		FMT_ROOT_PRINT("Zscored in {} sec\n", get_duration_s(stime, etime));
		FMT_PRINT_RT("tf_norms size: {}x{}, gene_norms size: {}x{}\n", tf_norms.rows(), tf_norms.columns(), gene_norms.rows(), gene_norms.columns());

		stime = getSysTime();
		output.resize(tf_norms.rows(), tf_norms.columns());
		
		if (app_params.method == app_parameters::method_type::CLR ) {
				using KernelType = wave::kernel::zscored_clr_kernel<double>;
				KernelType transform;
				splash::pattern::GlobalBinaryOp<MatrixType, MatrixType, 
					KernelType,
					MatrixType> transformer;
				transformer(tf_norms, gene_norms, transform, output);
		} else if (app_params.method == app_parameters::method_type::STOUFFER) {
				using KernelType = wave::kernel::zscored_stouffer_kernel<double>;
				KernelType transform;
				splash::pattern::GlobalBinaryOp<MatrixType, MatrixType, 
					KernelType,
					MatrixType> transformer;
				transformer(tf_norms, gene_norms, transform, output);
		}
		// output should be distributed.
		// do binary op with the 2.
		etime = getSysTime();
		FMT_ROOT_PRINT("transformed, with TF, in {} sec\n", get_duration_s(stime, etime));

		FMT_PRINT_RT("output size: {}x{}\n", output.rows(), output.columns());

	} else {
		stime = getSysTime();
		output.resize(input.rows(), input.columns());
		// set zscore degree of freedom to 1 to use unbiased variance
		using ReducType = splash::kernel::GaussianParamsExclude1<double, double, true>;
		ReducType reduc;
		if (app_params.method == app_parameters::method_type::CLR ) {
				using KernelType = wave::kernel::clr_vector_kernel<double>;
				KernelType transform;
				splash::pattern::GlobalReduceTransform<MatrixType, 
					ReducType, KernelType,
					MatrixType> transformer;
				transformer(input, reduc, transform, output);
		} else if (app_params.method == app_parameters::method_type::STOUFFER) {
				using KernelType = wave::kernel::stouffer_vector_kernel<double>;
				KernelType transform;
				splash::pattern::GlobalReduceTransform<MatrixType, 
					ReducType, KernelType,
					MatrixType> transformer;
				transformer(input, reduc, transform, output);
		}
		etime = getSysTime();
		FMT_ROOT_PRINT("Transformed in {} sec\n", get_duration_s(stime, etime));

	}
	// =============== WRITE OUT ==============
	// NOTE: rank 0 writes out.
	stime = getSysTime();
	if (app_params.tf_input.length() > 0) {
		FMT_PRINT_RT("writing {} tf_names, {} sample names, {}x{} output\n", tf_names.size(), samples.size(), output.rows(), output.columns());
		write_matrix_distributed(common_params.output, "array", tf_names, samples, output);
	} else {
		write_matrix_distributed(common_params.output, "array", genes, samples, output);
	}
	etime = getSysTime();
	FMT_ROOT_PRINT("Output in {} sec\n", get_duration_s(stime, etime));
	FMT_FLUSH();

	return 0;
}
