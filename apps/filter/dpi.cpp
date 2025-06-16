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
#include <fstream>
#include <iostream>
#include <vector>
#include <unordered_set>
#include <string>  //getline

#include "CLI/CLI.hpp"
#include "splash/io/CLIParserCommon.hpp"
#include "splash/io/parameters_base.hpp"
#include "splash/utils/benchmark.hpp"
#include "splash/utils/report.hpp"
#include "splash/ds/aligned_matrix.hpp"
#include "splash/patterns/pattern.hpp"

#include "mcp/filter/dpi.hpp"
#include "mcp/filter/threshold.hpp"

#include "splash/io/matrix_io.hpp"


#ifdef USE_OPENMP
#include <omp.h>
#endif

// TODO:	[X] why are the dpi values sometimes larger and sometimes smaller than the TF restricted ones?  cause is comparing wrong rows: gene-gene rows to tf-gen rows...
//			[X] ratio1, non-tf, yeast, has repeatable message truncated error. FIXED: seq matrix write was trying to mpi gather.
//			[X] test with all genes as TF:  missing TFs.  FIXED: h5 string storage missed trailing null char.
//			[X] verify that using all genes as TF is the same as not using TF.
//			[ ] speed for dpi_X is significantly worse than pure dpi.  one possibility is dpi has early stopping.  another is non-optimized compile.

// -------
//    re. maxmin value for TF filtered being larger than unfiltered.
// suppose pairwise min returns set MN = {a, b, c, d, e}.  WLoG, let max(MN) = d for unfiltered.
//		then d >= max(MN\d), by definition.
// with filtered, 2 cases:  d is filtered out, or not.
//		if d is filtered:   d >= d' = max(MN\d)
//		if d is not filtered:  d' = d, as all removed elements are <= d by definition.
// therefore, TF filtering should produce results that are <= than unfiltered results, for maxmin output.


template<typename DataType>
class app_parameters : public parameters_base {
	public:
		DataType tolerance;  // default value depends on usage.
        std::string tf_input;
		int tf_gene_transition;
		DataType diagonal;

		app_parameters(double const & tol = 0.1) : tolerance(tol), tf_gene_transition(-1), diagonal(0.0) {}
        app_parameters(const app_parameters<double>& other){
            tolerance = DataType(other.tolerance);
            tolerance = DataType(other.tolerance);
            tf_input = other.tf_input;
            tf_gene_transition = other.tf_gene_transition;
        }

		virtual ~app_parameters() {}

		virtual void config(CLI::App& app) {
			app.add_option("-l,--tolerance", tolerance, "value comparison tolerance.")->group("dpi")->check(CLI::Range(0.0, 1.0));
			auto tf_opt = app.add_option("-f,--tf-input", tf_input, "transcription factor file, 1 per line.")->group("transcription factor");
            auto transition_opt = app.add_option("--tf-gene-transition", tf_gene_transition, "transition from tf-tf to gene-gene. default at 0")->group("transcription factor")->check(CLI::Range(-1, 3));
			app.add_option("--diagonal", diagonal, "Input MI matrix diagonal should be set as this value. default 0. if negative, use original MI")->group("dpi");
			transition_opt->needs(tf_opt);
		}
		virtual void print(const char * prefix) const {
			FMT_ROOT_PRINT("{} DPI tolerance            : {}\n", prefix, tolerance); 
            FMT_ROOT_PRINT("{} TF input                 : {}\n", prefix, tf_input.c_str()); 
            FMT_ROOT_PRINT("{} TF-Gene transition level : {}\n", prefix, tf_gene_transition); 

			FMT_ROOT_PRINT("{} MI diagonal set to       : {}\n", prefix, diagonal); 

		}
        
};

template<typename DataType>
void run(splash::io::common_parameters& common_params,
         splash::io::mpi_parameters& mpi_params,
         app_parameters<DataType>& app_params ){

	// =============== SETUP INPUT ===================
	// NOTE: input data is replicated on all MPI procs.
	using MatrixType = splash::ds::aligned_matrix<DataType>;
	MatrixType input;
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
			common_params.vector_size, genes, samples,
			common_params.skip, 1, common_params.h5_gene_key,
            common_params.h5_samples_key, common_params.h5_matrix_key);
	}
	if(input.rows() == samples.size() && input.columns() == genes.size()){
		input = input.local_transpose();
	}
	etime = getSysTime();
	FMT_ROOT_PRINT("Load data in {} sec\n", get_duration_s(stime, etime));

	stime = getSysTime();
	size_t mc = std::min(input.rows(), input.columns());
	if (app_params.diagonal >= 0.0) {
		for (size_t i = 0; i < mc; ++i) {
			input.at(i, i) = app_params.diagonal;
		}
	}
	etime = getSysTime();
	FMT_ROOT_PRINT("set diag in {} sec\n", get_duration_s(stime, etime));

	MatrixType output;
	
	if (mpi_params.rank == 0) {
		mpi_params.print("[PARAM] ");
		common_params.print("[PARAM] ");
		app_params.print("[PARAM] ");
        std::cout << "No. GENES    : " << genes.size() << std::endl;
        std::cout << "No. SAMPLES  : " << samples.size() << std::endl;
        std::cout << "INPUT SIZE   : " << input.rows() << "x" << input.columns() << std::endl;
	}
	if(input.rows() != genes.size() && input.columns() != samples.size()){
		if(mpi_params.rank == 0) {
            std::cout << "INPUT SIZE   : "
            	<< input.rows() << " x " << input.columns()
            	<< "DOEST NOT MATCH genes X samples "
            	<< genes.size() << "x" << samples.size()
            	<< std::endl;
		}
		return;
	}

	// =============== PARTITION and RUN ===================
	assert((input.rows() == input.columns()) && "input matrix should be square");
	// FMT_PRINT("genes {}, samples {}\n", genes.size(), samples.size());

	std::unordered_set<std::string> TFs;
	std::unordered_set<std::string> gns(genes.begin(), genes.end());
	std::vector<DataType> tfs;
	std::vector<std::string> tf_names;
	if (app_params.tf_input.length() > 0) {
		stime = getSysTime();

		// read the file and put into a set.
		std::ifstream ifs(app_params.tf_input);
		std::string line;
		// FMT_ROOT_PRINT("TF file: ");
		while (std::getline(ifs, line)) {
			TFs.insert(splash::utils::trim(line));
			// FMT_ROOT_PRINT("{},", splash::utils::trim(line));
		}
		// FMT_ROOT_PRINT("\n   total {}\n", TFs.size());
		ifs.close();
	
		// now check for each gene whether it's in TF list.
		for (size_t i = 0; i < genes.size(); ++i) {
			DataType res = TFs.count(splash::utils::trim(genes[i])) > 0 ? std::numeric_limits<DataType>::max() : std::numeric_limits<DataType>::lowest();
			// FMT_PRINT("\"{}\"\t{}\t{}\n", splash::utils::trim(genes[i]), genes[i].length(), (res ? "yes" : "no"));
			tfs.push_back(res); // if this is one of the target TF, then safe it
			if (res >= 0.0) {
				tf_names.push_back(genes[i]);
			}
		}

		etime = getSysTime();
		FMT_ROOT_PRINT("Load TF data in {} sec\n", get_duration_s(stime, etime));
		if (mpi_params.rank == 0) {
			FMT_ROOT_PRINT("Selected TFs {}:", tf_names.size());
			for (auto TF : tf_names) {
				FMT_ROOT_PRINT("{},", TF.c_str());
			}
			FMT_ROOT_PRINT("\n");
			FMT_ROOT_PRINT("MISSING TFs:");
			for (auto TF : TFs) {
				if (gns.count(splash::utils::trim(TF)) == 0)
					FMT_ROOT_PRINT("{},", TF.c_str());
			}
			FMT_ROOT_PRINT("\n");
		}
		FMT_ROOT_PRINT("TFs specified {}, found {}\n", TFs.size(), tf_names.size());

		if (tf_names.size() == 0) {
			return;
		}
	}

	//======== NOT making smaller arrays.
	MatrixType tfs_to_genes(1, input.columns());    // tfs as rows
	MatrixType genes_to_genes = input;
	if (app_params.tf_input.length() > 0) {
		
		// select rows.
		tfs_to_genes.resize(tf_names.size(), input.columns());
		for (size_t i = 0, j = 0; i < genes.size(); ++i) {
			if (tfs[i] > 0) {
				memcpy(tfs_to_genes.data(j), input.data(i), input.column_bytes());
				
				++j;
			}
		}

	}

	
	// =============== PARTITION and RUN ===================
	stime = getSysTime();

	using MaskType = splash::ds::aligned_matrix<bool>;
	MaskType mask(input.rows(), input.columns());

	// correlation close to 0 is bad.
	if (tfs.size() > 0) { // using pvalue threshold
		using KernelType = mcp::kernel::dpi_tf_kernel<DataType>;
		KernelType kernel(tfs, app_params.tolerance);
		using MaskGenType = ::splash::pattern::InnerProduct<MatrixType, KernelType, MaskType>;
		MaskGenType maskgen;
		maskgen(input, input, kernel, mask);
		
	} else {
		using KernelType = mcp::kernel::dpi_kernel<DataType>;
		KernelType kernel(app_params.tolerance);
		using MaskGenType = ::splash::pattern::InnerProduct<MatrixType, KernelType, MaskType>;
		MaskGenType maskgen;
		maskgen(input, input, kernel, mask);
	}

	etime = getSysTime();
	FMT_PRINT_RT("Generate Mask ({} x {}) in {} sec\n", mask.rows(), mask.columns(), get_duration_s(stime, etime));

	// stime = getSysTime();
	// write_matrix_distributed("mask.h5", "array", genes, samples, mask);
	// etime = getSysTime();
	// FMT_ROOT_PRINT("Mask Output in {} sec\n", get_duration_s(stime, etime));
	// FMT_FLUSH();
	// =============== now mask it ===================
	stime = getSysTime();

	// mask should be split at this point.
	size_t rows = mask.rows();
	size_t offset = 0;
#ifdef USE_MPI
	MPI_Exscan(&rows, &offset, 1, MPI_UNSIGNED_LONG, MPI_SUM, MPI_COMM_WORLD);
	if (mpi_params.rank == 0) offset = 0;
#endif

	MatrixType dout; 
	size_t removed = 0;
	{
		using KernelType = mcp::kernel::mask<DataType, true>;
		KernelType kernel;
		using MaskerType = ::splash::pattern::BinaryOp<MatrixType, MaskType, KernelType, MatrixType>;
		MaskerType masker;
		splash::utils::partition<size_t> in_part(offset, rows, mpi_params.rank);
		splash::utils::partition<size_t> out_part(0, rows, mpi_params.rank);
		dout.resize(rows, input.columns());

		masker(input, in_part, mask, out_part, kernel, dout, out_part);
		removed = masker.processed;
	}    
	
	size_t local_rows = 0;
	size_t imax = offset + rows;
	if (tfs.size() > 0) {
		// count
		for (size_t i = offset; i < imax; ++i) {
			local_rows += (tfs[i] > 0);
		}
		// select rows.
		output.resize(local_rows, input.columns());
		for (size_t i = 0, j = 0; i < rows; ++i) {
			if (tfs[i + offset] > 0) {
				memcpy(output.data(j), dout.data(i), dout.column_bytes());
				++j;
			}
		}
	} else {
		output = dout;
	}

	etime = getSysTime();
	FMT_ROOT_PRINT("Applied Mask in {} sec\n", get_duration_s(stime, etime));
	FMT_ROOT_PRINT("Mask removing {} elements \n", removed);


	// =============== WRITE OUT ==============
	// NOTE: rank 0 writes out.
	stime = getSysTime();
	FMT_PRINT("output size: {} x {}\n", output.rows(), output.columns());
	if (app_params.tf_input.length() > 0) {
		write_matrix_distributed(common_params.output, "array", tf_names, samples, output);
	} else {
		write_matrix_distributed(common_params.output, "array", genes, samples, output);
	}
	etime = getSysTime();
	FMT_ROOT_PRINT("Output in {} sec\n", get_duration_s(stime, etime));
	FMT_FLUSH();


}



int main(int argc, char* argv[]) {

	//==============  PARSE INPUT =====================
	CLI::App app{"Data Processing Inequality"};

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
	auto stime = getSysTime();
	auto etime = getSysTime();

    if(common_params.use_single) {
        app_parameters<float> flapp_params(app_params);
        run<float>(common_params, mpi_params, flapp_params);
    } else {
        run<double>(common_params, mpi_params, app_params);
    }
	etime = getSysTime();
	FMT_ROOT_PRINT("Total DPI computation in {} sec\n", get_duration_s(stime, etime));
	
	return 0;
}
