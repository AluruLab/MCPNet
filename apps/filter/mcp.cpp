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

#include "mcp/filter/mcp.hpp"
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


class app_parameters : public parameters_base {
	public:
		enum method_type : int { 
			MCP2 = 1, 
			MCP3 = 2, 
			MCP4 = 3, 
			BIN_MCP = 5};

		method_type compute;
        std::string tf_input;
		std::string first_in;
		std::string second_in;
		int tf_gene_transition;
		bool clamped;
		std::vector<double> diagonals;
		std::vector<std::string> maxmin_outputs;

		app_parameters(double const & tol = 0.1) : compute(MCP2), tf_gene_transition(-1), clamped(false) {}
		virtual ~app_parameters() {}

		virtual void config(CLI::App& app) {
			auto tf_opt = app.add_option("-f,--tf-input", tf_input, "transcription factor file, 1 per line.")->group("transcription factor");
            auto transition_opt = app.add_option("--tf-gene-transition", tf_gene_transition, "transition from tf-tf to gene-gene. default at 0")->group("transcription factor")->check(CLI::Range(-1, 3));
			transition_opt->needs(tf_opt);

			auto comp_opt = app.add_option("-m,--method", compute, "Algorithm: MCP2=1, MCP3=2, MCP4=3, MCP from Partials=5")->group("MCP");

			app.add_option("--first", first_in, "first input, for computing MCP from partials.")->group("MCP");
			app.add_option("--second", second_in, "second input, for computing MCP from partials.")->group("MCP");
			app.add_option("--diagonal", diagonals, "Input MI matrix diagonal should be set as this value. default 0. if negative, use original MI")->group("MCP");
            			
			auto clamped_opt = app.add_flag("--clamped", clamped, "output is clampped")->group("MCP");
			clamped_opt->needs(comp_opt);

			auto maxmin_opt = app.add_option("-x,--maxmin-output", maxmin_outputs, "filename for maxmin intermediate values (1 or more files)")->group("MCP");
			maxmin_opt->needs(comp_opt);


		}
		virtual void print(const char * prefix) {
            FMT_ROOT_PRINT("{} MCP compute method: {}\n", prefix, 
				(compute == MCP2 ? "MCP2" : 
				(compute == MCP3 ? "MCP3" : 
				(compute == MCP4 ? "MCP4" : 
				(compute == BIN_MCP ? "" : 
				"unknown"))))); 
				// ); 
            FMT_ROOT_PRINT("{} first input: {}\n", prefix, first_in.c_str()); 
			FMT_ROOT_PRINT("{} second input: {}\n", prefix, second_in.c_str()); 

            FMT_ROOT_PRINT("{} TF input: {}\n", prefix, tf_input.c_str()); 
            FMT_ROOT_PRINT("{} TF-Gene transition: level {}\n", prefix, tf_gene_transition); 

			for (auto maxmin_out : maxmin_outputs) {
	            FMT_ROOT_PRINT("{} MaxMin intermediate output: {}\n", prefix, maxmin_out.c_str()); 
			}

			FMT_ROOT_PRINT("{} MCP compute clamping output: {}\n", prefix, (clamped ? "true" : "false")); 
			for (auto d : diagonals) {
				FMT_ROOT_PRINT("{} MI diagonal set to: {}\n", prefix, d); 
			}

		}
        
};



int main(int argc, char* argv[]) {

	//==============  PARSE INPUT =====================
	CLI::App app{"Data Processing Inequality"};

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
	MatrixType input, input1, input2;
	std::vector<std::string> genes, genes1, dummy1, genes2;
	std::vector<std::string> samples;
	size_t input1_x, input2_y, dummy2;
	
	auto stime = getSysTime();
	auto etime = getSysTime();

	stime = getSysTime();
	if (common_params.random) {
		input = make_random_matrix(common_params.rseed, 
			common_params.rmin, common_params.rmax, 
			common_params.num_vectors, common_params.vector_size,
			genes, samples);

		if(app_params.compute == app_parameters::method_type::BIN_MCP) {
			input1 = make_random_matrix(common_params.rseed + 2, 
				common_params.rmin, common_params.rmax, 
				common_params.num_vectors, common_params.num_vectors);
			input2 = make_random_matrix(common_params.rseed + 8, 
				common_params.rmin, common_params.rmax, 
				common_params.num_vectors, common_params.num_vectors);
		}

	} else {
		input = read_matrix<double>(common_params.input, "array", 
			common_params.num_vectors, common_params.vector_size,
			genes, samples);

		input1_x = input2_y = dummy2 = common_params.num_vectors;

		if(app_params.compute == app_parameters::method_type::BIN_MCP) {
			input1 = read_matrix<double>(app_params.first_in, "array", 
				input1_x, dummy2, genes1, dummy1);
			input2 = read_matrix<double>(app_params.second_in, "array", 
				dummy2, input2_y, dummy1, genes2);;
		}
	}
	etime = getSysTime();
	FMT_ROOT_PRINT("Load data in {} sec\n", get_duration_s(stime, etime));

	stime = getSysTime();
	size_t mc = std::min(input.rows(), input.columns());
	// if diagonal is specified, then use it.
	bool reset_diag = app_params.diagonals.size() > 0;
	if (reset_diag) {
		double diagonal = app_params.diagonals[0];
		for (size_t i = 0; i < mc; ++i) {
			input.at(i, i) = diagonal;
		}
		mc = std::min(input1.rows(), input1.columns());
		for (size_t i = 0; i < mc; ++i) {
			input1.at(i, i) = diagonal;
		}
		mc = std::min(input2.rows(), input2.columns());
		for (size_t i = 0; i < mc; ++i) {
			input2.at(i, i) = diagonal;
		}
	}
	etime = getSysTime();
	FMT_ROOT_PRINT("set diag in {} sec\n", get_duration_s(stime, etime));

	MatrixType output;
	
	if (mpi_params.rank == 0) {
		mpi_params.print("[PARAM] ");
		common_params.print("[PARAM] ");
		app_params.print("[PARAM] ");
	}

	// =============== PARTITION and RUN ===================
	assert((input.rows() == input.columns()) && "input matrix should be square");
	// FMT_PRINT("genes {}, samples {}\n", genes.size(), samples.size());

	std::unordered_set<std::string> TFs;
	std::unordered_set<std::string> gns(genes.begin(), genes.end());
	std::vector<double> tfs;
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
			double res = TFs.count(splash::utils::trim(genes[i])) > 0 ? std::numeric_limits<double>::max() : std::numeric_limits<double>::lowest();
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
			return 1;
		}
	}

	//======== NOT making smaller arrays.
	MatrixType tfs_to_genes(1, input.columns());    // tfs as rows
	// MatrixType genes_to_tfs(input.rows(), 1);    // tfs as columns.  This is replaced with column masking on full matrix.
	// MatrixType tfs_to_tfs(1, 1);		// tfs as columns  This is replaced with column masking on tfs-gene matrix.
	MatrixType genes_to_genes = input;
	if (app_params.tf_input.length() > 0) {
		
		// select rows.
		tfs_to_genes.resize(tf_names.size(), input.columns());
		// genes_to_tfs.resize(input.rows(), tf_names.size());
		// tfs_to_tfs.resize(tf_names.size(), tf_names.size());
		for (size_t i = 0, j = 0; i < genes.size(); ++i) {
			if (tfs[i] > 0) {
				memcpy(tfs_to_genes.data(j), input.data(i), input.column_bytes());

				++j;
			}
		}

		// // write out TF->gene MI if there is one
		// if ((app_params.maxmin_outputs.size() > 0) && (app_params.maxmin_outputs[0].length() > 0)) {
		// 	if (mpi_params.rank == 0) {
		// 		stime = getSysTime();
		// 		write_matrix(app_params.maxmin_outputs[0], "array", tf_names, samples, tfs_to_genes);
		// 		etime = getSysTime();
		// 	}
		// 	FMT_ROOT_PRINT("TF-GENE MI Output in {} sec\n", get_duration_s(stime, etime));
		// }


	}

	// the approach of using tf-tf, tf-gene, and gene-gene arrays, and inner products with position aware kernels
	// is troublesome.
	//		1. I(x,x) still need to be zero in the output.  in tf-gene matrix this is no longer clear as the "tf" loses the original x info,
	//      2. dpi has a check for xz to be 0.  z range may exceed tf-tf column dimensions.
	//   this means we need to pass more info into the kernels.

	// consider - full, symmetric, array, with nonTF columns set 0's.  vector min would put non-tf array elements at 0 and they would not affect the max.
	//   this produces a full matrix but only uses TFs?
	// gene-gene-gene:  full MI to full MI
	// tf-gene-gene:    TF rows to full MI.  non-tf are 0.  can post filter from gen-gen-gen.
	// tf-tf-gene: 		TF rows with TF col, to full MI.  nonTF rows get 0.  nonTF cols do not contribute for each gene.
	// USING MASKS for columns.  Use TF to gene array for rows.

	// weighted average.
	// =============== PARTITION and RUN ===================
	// leg -1: g-g-g (gg to gg), g-g-g-g (gg to ggg), g-g-g-g-g (ggg to ggg)
	// leg 0 : t-g-g (tg to gg), t-g-g-g (tg to ggg), t-g-g-g-g (tgg to ggg)
	// leg 1 : t-t-g (tg to gg, masking), t-t-g-g (tg to ggg, masking), t-t-g-g-g (ttg to ggg)
	// leg 2 : t-t-g (tg to gg, masking), t-t-t-g (tg to gtg (gg to gg, masking), masking), t-t-t-g-g (ttg to ggg, masking)  
	// leg 3 : t-t-g (tg to gg, masking), t-t-t-g (tg to gtg (gg to gg, masking), masking), t-t-t-t-g (ttg to gtg, masking)
	// need ggg, tgg, ttg, gtg,
	// 	need both kernel types.
	//  need symmetric and assymetric patterns for both types.

	if (app_params.compute == app_parameters::method_type::MCP2) {
		stime = getSysTime();

		// first max{min}
		MatrixType dmaxmin1; 
		if (app_params.tf_input.length() > 0) {
			dmaxmin1.resize(tf_names.size(), input.columns());
			if (app_params.tf_gene_transition == 1) {
				// tfs-tfs-genes
				using MXKernelType = mcp::kernel::mcp2_maxmin_kernel<double, true, true>;
				MXKernelType mxkernel(tfs);  // clamped.
				using MXGenType = ::splash::pattern::InnerProduct<MatrixType, MXKernelType, MatrixType, false>;
				MXGenType mxgen;
				// tf-gen transition on second gene.
				mxgen(tfs_to_genes, genes_to_genes, mxkernel, dmaxmin1);
			} else {
				// tfs-genes-genes
				using MXKernelType = mcp::kernel::mcp2_maxmin_kernel<double, false, true>;
				MXKernelType mxkernel;  // clamped.
				using MXGenType = ::splash::pattern::InnerProduct<MatrixType, MXKernelType, MatrixType, false>;
				MXGenType mxgen;
				// default, edge has tf to gene transition.
				mxgen(tfs_to_genes, genes_to_genes, mxkernel, dmaxmin1);
			}

		} else {
			using MXKernelType = mcp::kernel::mcp2_maxmin_kernel<double, false, true>;
			MXKernelType mxkernel;  // clamped.
			using MXGenSType = ::splash::pattern::InnerProduct<MatrixType, MXKernelType, MatrixType, true>;
			MXGenSType mxgen;
			dmaxmin1.resize(genes_to_genes.rows(), genes_to_genes.columns());
			mxgen(genes_to_genes, genes_to_genes, mxkernel, dmaxmin1);
		}
		etime = getSysTime();
		FMT_ROOT_PRINT("Compute max of row-wise min in {} sec\n", get_duration_s(stime, etime));

		if ((app_params.maxmin_outputs.size() > 0) && (app_params.maxmin_outputs[0].length() > 0)) {
			stime = getSysTime();
			if (app_params.tf_input.length() > 0)
				write_matrix_distributed(app_params.maxmin_outputs[0], "array", tf_names, samples, dmaxmin1);
			else 
				write_matrix_distributed(app_params.maxmin_outputs[0], "array", genes, samples, dmaxmin1);
			etime = getSysTime();
			FMT_ROOT_PRINT("MaxMin Output in {} sec\n", get_duration_s(stime, etime));
		}

		stime = getSysTime();
		MatrixType maxmin1 = dmaxmin1.allgather();

		// correlation close to 0 is bad.
		using KernelType = mcp::kernel::ratio_kernel<double, double>;
		KernelType kernel;  // clamped.
		using TolGenType = ::splash::pattern::GlobalBinaryOp<MatrixType, MatrixType, KernelType, MatrixType>;
		TolGenType tolgen;
		if (app_params.tf_input.length() > 0) {
			tolgen(tfs_to_genes, maxmin1, kernel, output);
		} else {
			tolgen(genes_to_genes, maxmin1, kernel, output);
		}
		etime = getSysTime();
		FMT_ROOT_PRINT("Compute MI Ratio in {} sec\n", get_duration_s(stime, etime));


	} else if (app_params.compute == app_parameters::method_type::MCP3) {
		// =============== PARTITION and RUN ===================
		stime = getSysTime();

		using MaskedMXKernelType = mcp::kernel::mcp2_maxmin_kernel<double, true, true>;
		MaskedMXKernelType maskedmxkernel(tfs);  // clamped.
		using MXKernelType = mcp::kernel::mcp2_maxmin_kernel<double, false, true>;
		MXKernelType mxkernel;  // clamped.

		// first max{min}

		// compute leg2 to 3.  OUTPUT MUST BE input.rows() x input.columns().  Potentially skipping rows. 
		MatrixType dmaxmin1; 
		dmaxmin1.resize(genes_to_genes.rows(), genes_to_genes.columns());
		if ((app_params.tf_input.length() > 0) && (app_params.tf_gene_transition == 2)) {  // leg 2.
			using MXGenType = ::splash::pattern::InnerProduct<MatrixType, MaskedMXKernelType, MatrixType, true>;
			MXGenType mxgen;
			// tf-(tf-tf-gen) transition.  make gen-tf-gen.
			mxgen(genes_to_genes, genes_to_genes, maskedmxkernel, dmaxmin1);
		} else {  // leg 0, 1, and none.
			using MXGenType = ::splash::pattern::InnerProduct<MatrixType, MXKernelType, MatrixType, true>;
			MXGenType mxgen;
			// tf-(tf-gen-gen), tf-(gene-gene-gene), gene-(gene-gene-gene)  make gene-gene-gene.  
			mxgen(genes_to_genes, genes_to_genes, mxkernel, dmaxmin1);
		}
		MatrixType maxmin1 = dmaxmin1.allgather();
		etime = getSysTime();
		FMT_ROOT_PRINT("Compute max of row-wise min, pass 1 in {} sec\n", get_duration_s(stime, etime));

		stime = getSysTime();
		// second max{min}
		MatrixType dmaxmin2;
		if (app_params.tf_input.length() > 0) {
			dmaxmin2.resize(tfs_to_genes.rows(), maxmin1.columns());
			if ((app_params.tf_gene_transition == 2) || (app_params.tf_gene_transition == 1)) {
				// tf-tf-gene-gene, or tf-tf-tf-gene
				using MXGenType = ::splash::pattern::InnerProduct<MatrixType, MaskedMXKernelType, MatrixType, false>;
				MXGenType mxgen;
				mxgen(tfs_to_genes, maxmin1, maskedmxkernel, dmaxmin2);
			} else {  // leg 0.
				// tf-(gene-gene-gene), edge has tf to gene transition.
				using MXGenType = ::splash::pattern::InnerProduct<MatrixType, MXKernelType, MatrixType, false>;
				MXGenType mxgen;
				mxgen(tfs_to_genes, maxmin1, mxkernel, dmaxmin2);
			}
		} else {
			// genes-genes-genes-genes
			dmaxmin2.resize(genes_to_genes.rows(), maxmin1.columns());
			using MXGenSType = ::splash::pattern::InnerProduct<MatrixType, MXKernelType, MatrixType, false>;
			MXGenSType mxgen_s;
			mxgen_s(genes_to_genes, maxmin1, mxkernel, dmaxmin2);  // NOTE this is not symmetric? previous logic had that assumption.
		}
		MatrixType maxmin2 = dmaxmin2.allgather();
		etime = getSysTime();
		FMT_ROOT_PRINT("Compute  max of row-wise min, pass 2 in {} sec\n", get_duration_s(stime, etime));

		// transpose and then do max IF SYMMETRIC
		stime = getSysTime();
		MatrixType maxmat;
		if (app_params.tf_input.length() == 0) {
			// second max{min}
			MatrixType maxmin2t = maxmin2.local_transpose(); 

			// do max.
			using MaxKernelType = mcp::kernel::max_kernel<double>;
			MaxKernelType maxkernel;  // clamped.
			using MaxGenType = ::splash::pattern::BinaryOp<MatrixType, MatrixType, MaxKernelType, MatrixType>;
			MaxGenType maxgen;
			maxmat.resize(maxmin2.rows(), maxmin2.columns());
			maxgen(maxmin2, maxmin2t, maxkernel, maxmat);
		} else {
			maxmat = maxmin2;
		}
		etime = getSysTime();
		FMT_ROOT_PRINT("Compute max(A, A') in {} sec\n", get_duration_s(stime, etime));

		if ((mpi_params.rank == 0) && (app_params.maxmin_outputs.size() > 0) && (app_params.maxmin_outputs[0].length() > 0)) {
			stime = getSysTime();
			if (app_params.tf_input.length() > 0) {
				write_matrix(app_params.maxmin_outputs[0], "array", tf_names, samples, maxmat);
			} else {
				write_matrix(app_params.maxmin_outputs[0], "array", genes, samples, maxmat);
			}
			etime = getSysTime();
			FMT_ROOT_PRINT("MaxMin Output (1 core) in {} sec\n", get_duration_s(stime, etime));
		}

		stime = getSysTime();
		// correlation close to 0 is bad.
		using KernelType = mcp::kernel::ratio_kernel<double, double>;
		KernelType kernel;  // clamped.
		using TolGenType = ::splash::pattern::GlobalBinaryOp<MatrixType, MatrixType, KernelType, MatrixType>;
		TolGenType tolgen;
		if (app_params.tf_input.length() > 0) {
			tolgen(tfs_to_genes, maxmat, kernel, output);
		} else {
			tolgen(genes_to_genes, maxmat, kernel, output);
		}

		etime = getSysTime();
		FMT_ROOT_PRINT("Compute MI Ratios in {} sec\n", get_duration_s(stime, etime));


	} else if (app_params.compute == app_parameters::method_type::MCP4) {
		// =============== PARTITION and RUN ===================
		stime = getSysTime();

		using MaskedMXKernelType = mcp::kernel::mcp2_maxmin_kernel<double, true, true>;
		MaskedMXKernelType maskedmxkernel(tfs);  // clamped.
		using MXKernelType = mcp::kernel::mcp2_maxmin_kernel<double, false, true>;
		MXKernelType mxkernel;  // clamped.
		using MXGenSType = ::splash::pattern::InnerProduct<MatrixType, MXKernelType, MatrixType, true>;
		MXGenSType mxgen_s;

		// first max{min}
		MatrixType dmaxmin1;
		dmaxmin1.resize(genes_to_genes.rows(), genes_to_genes.columns());
		if ((app_params.tf_input.length() > 0) && (app_params.tf_gene_transition == 3)) {
			// tf-tf-(tf-tf-gene) and 
			// tf-gen transition on second gene.
			using MXGenType = ::splash::pattern::InnerProduct<MatrixType, MaskedMXKernelType, MatrixType, true>;
			MXGenType mxgen;
			mxgen(genes_to_genes, genes_to_genes, maskedmxkernel, dmaxmin1);

		} else {
			// tf-tf-(tf-gene-gene), tf-tf-(gene-gene-gene) tf-gene-(gene-gene-gene), gene-gene-(gene-gene-gene)
			mxgen_s(genes_to_genes, genes_to_genes, mxkernel, dmaxmin1);
		}
		MatrixType maxmin1 = dmaxmin1.allgather();
		etime = getSysTime();
		FMT_ROOT_PRINT("Compute max of row-wise min, part1 1 in {} sec\n", get_duration_s(stime, etime));

		stime = getSysTime();
		// second max{min}
		MatrixType dmaxmin2;
		if (app_params.tf_input.length() > 0) {
			dmaxmin2.resize(tfs_to_genes.rows(), genes_to_genes.columns());

			if (app_params.tf_gene_transition == 0) { 
				// (tf-gene-gene)-gene-gene on leg 0.
				using MXGenType = ::splash::pattern::InnerProduct<MatrixType, MXKernelType, MatrixType, false>;
				MXGenType mxgen;
				mxgen(tfs_to_genes, genes_to_genes, mxkernel, dmaxmin2);
			} else {
				using MXGenType = ::splash::pattern::InnerProduct<MatrixType, MaskedMXKernelType, MatrixType, false>;
				MXGenType mxgen;
				// (tf-tf-tf)-tf-gene  (tf-tf-tf)-gene-gene), (tf-tf-gene)-gene-gene
				mxgen(tfs_to_genes, genes_to_genes, maskedmxkernel, dmaxmin2);
			}
		} else {
			dmaxmin2.resize(genes_to_genes.rows(), genes_to_genes.columns());
			// (gene-gene-gene)-gene-gene
			mxgen_s(genes_to_genes, genes_to_genes, mxkernel, dmaxmin2);
		}
		MatrixType maxmin2 = dmaxmin2.allgather();
		etime = getSysTime();
		FMT_ROOT_PRINT("Compute  max of row-wise min, part 2 in {} sec\n", get_duration_s(stime, etime));

		stime = getSysTime();
		MatrixType dmaxmin3;
		if (app_params.tf_input.length() > 0) {
			dmaxmin3.resize(tfs_to_genes.rows(), genes_to_genes.columns());
			if ((app_params.tf_gene_transition == 2) || (app_params.tf_gene_transition == 3)) {  // leges  2, 3
				using MXGenType = ::splash::pattern::InnerProduct<MatrixType, MaskedMXKernelType, MatrixType, false>;
				MXGenType mxgen;
				// (tf-tf-tf)-tf-gene  (tf-tf-tf)-gene-gene), 
				mxgen(maxmin2, maxmin1, maskedmxkernel, dmaxmin3);
			} else { // leg 0
				// (tf-tf-gene)-gene-gene (tf-gene-gene)-gene-gene on leg 0.
				using MXGenType = ::splash::pattern::InnerProduct<MatrixType, MXKernelType, MatrixType, false>;
				MXGenType mxgen;
				mxgen(maxmin2, maxmin1, mxkernel, dmaxmin3);
			}

		} else {
			dmaxmin3.resize(genes_to_genes.rows(), genes_to_genes.columns());
			// gene-gene-gene-gene-gene
			mxgen_s(maxmin2, maxmin1, mxkernel, dmaxmin3);
		}
		etime = getSysTime();
		FMT_ROOT_PRINT("Compute  max of row-wise min, part 3 in {} sec\n", get_duration_s(stime, etime));

		if ((app_params.maxmin_outputs.size() > 0) && (app_params.maxmin_outputs[0].length() > 0)) {
			stime = getSysTime();
			
			if (app_params.tf_input.length() > 0) {
				write_matrix_distributed(app_params.maxmin_outputs[0], "array", tf_names, samples, dmaxmin3);
			} else {
				write_matrix_distributed(app_params.maxmin_outputs[0], "array", genes, samples, dmaxmin3);
			}
			etime = getSysTime();
			FMT_ROOT_PRINT("MaxMin Output in {} sec\n", get_duration_s(stime, etime));
		}

		stime = getSysTime();
		MatrixType maxmin3 = dmaxmin3.allgather();

		// correlation close to 0 is bad.
		using KernelType = mcp::kernel::ratio_kernel<double, double>;
		KernelType kernel;  // clamped.
		using TolGenType = ::splash::pattern::GlobalBinaryOp<MatrixType, MatrixType, KernelType, MatrixType>;
		TolGenType tolgen;
		if (app_params.tf_input.length() > 0) {
			tolgen(tfs_to_genes, maxmin3, kernel, output);
		} else {
			FMT_PRINT_RT("maxmin3 dim: {}x{}\n", maxmin3.rows(), maxmin3.columns());
			tolgen(genes_to_genes, maxmin3, kernel, output);
		}

		etime = getSysTime();
		FMT_ROOT_PRINT("Compute MI Ratios in {} sec\n", get_duration_s(stime, etime));



	} else if (app_params.compute == app_parameters::method_type::BIN_MCP) {
		if (app_params.tf_input.length() > 0) {
			FMT_ROOT_PRINT("transcription factors are not supported when computing MCP score from two partial scores.\n");
			return 1;
		}

		stime = getSysTime();

		// first max{min}
		MatrixType dmaxmin1; 
		dmaxmin1.resize(input1.rows(), input2.rows());
		using MXKernelType = mcp::kernel::mcp2_maxmin_kernel<double, false, true>;
		MXKernelType mxkernel;  // clamped.
		using MXGenSType = ::splash::pattern::InnerProduct<MatrixType, MXKernelType, MatrixType, false>;
		MXGenSType mxgen;
		mxgen(input1, input2, mxkernel, dmaxmin1);
		etime = getSysTime();
		FMT_ROOT_PRINT("Compute max of row-wise min in {} sec\n", get_duration_s(stime, etime));

		if ((app_params.maxmin_outputs.size() > 0) && (app_params.maxmin_outputs[0].length() > 0)) {
			stime = getSysTime();
			write_matrix_distributed(app_params.maxmin_outputs[0], "array", genes, samples, dmaxmin1);
			etime = getSysTime();
			FMT_ROOT_PRINT("MaxMin Output in {} sec\n", get_duration_s(stime, etime));
		}

		stime = getSysTime();
		MatrixType maxmin1 = dmaxmin1.allgather();

		// correlation close to 0 is bad.
		using KernelType = mcp::kernel::ratio_kernel<double, double>;
		KernelType kernel;  // clamped.
		using TolGenType = ::splash::pattern::GlobalBinaryOp<MatrixType, MatrixType, KernelType, MatrixType>;
		TolGenType tolgen;
		tolgen(genes_to_genes, maxmin1, kernel, output);
		etime = getSysTime();
		FMT_ROOT_PRINT("Compute MI Ratio in {} sec\n", get_duration_s(stime, etime));
	}

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


	return 0;
}
