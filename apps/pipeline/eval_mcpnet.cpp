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
#include <iomanip> // ostringstream
#include <vector>
#include <unordered_set>
#include <string>  //getline
#include <random>

#include "CLI/CLI.hpp"
#include "splash/io/CLIParserCommon.hpp"
#include "splash/io/parameters_base.hpp"
#include "splash/utils/benchmark.hpp"
#include "splash/utils/report.hpp"
#include "splash/ds/aligned_matrix.hpp"
#include "splash/patterns/pattern.hpp"

#include "mcp/correlation/BSplineMI.hpp"
#include "mcp/correlation/AdaptivePartitioningMI.hpp"
#include "splash/transform/rank.hpp"

#include "mcp/filter/mcp.hpp"
#include "mcp/filter/threshold.hpp"
#include "mcp/transform/stouffer.hpp"
#include "mcp/transform/combine.hpp"
#include "splash/transform/zscore.hpp"
#include "mcp/transform/aupr.hpp"
#include "mcp/transform/auroc.hpp"

#include "splash/io/matrix_io.hpp"


#ifdef USE_OPENMP
#include <omp.h>
#endif

#include "mcpnet_helper.hpp"


// combined, end to end, for generation and testing.
// this imple avoids file IO.

class app_parameters : public parameters_base {
	public:
		enum method_type : int { 
			UNUSED = 0, 
			MCP2 = 1, 
			MCP3 = 2, 
			MCP4 = 3, 
			MU_MCP = 4,
			MU_MCP_STOUFFER = 5};
		enum mi_method_type : int {
			BSpline = 0,
			AP = 1
		};

		const int num_bins = 10;
		std::vector<method_type> computes;
		mi_method_type mi_computes;
		std::string mi_file;
        std::string coeffs;
		std::string groundtruth_list;
		std::string groundtruth_mat;
        std::string tf_input;
		bool clamped;
		double diagonal;
		bool combo_from_train;
		size_t iters;

		app_parameters() : mi_computes(mi_method_type::BSpline), clamped(false), diagonal(0.0), combo_from_train(true), iters(100) {}
		virtual ~app_parameters() {}

		virtual void config(CLI::App& app) {
			auto mi_group = app.add_option_group("MI");
			mi_group->add_option(app.add_option("--mi-method", mi_computes, "MI Algorithm: 0 = Bspline, 1 (default) = Adaptive Partitioning"));
			mi_group->add_option(app.add_option("--mi-file", mi_file, "precomputed MI file"));
			mi_group->require_option(1);

			auto comp_opt = app.add_option("-m,--method", computes, "Algorithm: MCP2=1, MCP3=2, MCP4=3, EnsembleMCP=4, EnsembleMCP with Stouffer=5")->group("MCP");
			auto clamped_opt = app.add_flag("--clamped", clamped, "output is clampped")->group("MCP");
			clamped_opt->needs(comp_opt);
			app.add_option("-f,--coefficients", coeffs, "file with combo coefficients.  For ensemble MCP only.")->group("MCP");

			auto gt_list_opt = app.add_option("-g,--groundtruth-list", groundtruth_list, "filename of groundtruth edge list.  require just forward or reverse to be present. (+) -> 1, (-) -> 0, unknown -> -1");
			auto gt_mat_opt = app.add_option("-x,--groundtruth-matrix", groundtruth_mat, "filename of groundtruth matrix. require symmetric matrix. (+) -> 1, (-) -> 0, unknown -> -1");
			app.add_option("--diagonal", diagonal, "Input MI matrix diagonal should be set as this value. default 0. if negative, use original MI")->group("MCP");

			app.add_option("--tf-input", tf_input, "transcription factor file, 1 per line.")->group("transcription factor");

			app.add_flag("--permute-combo", combo_from_train, "combo is optimized from permuted training groundtruth");
			app.add_option("--iterations", iters, "number of permutations for training/test evaluation");

			auto opt_group = app.add_option_group("evaluation");
            opt_group->add_option(gt_list_opt);
            opt_group->add_option(gt_mat_opt);
            opt_group->require_option(1);

		}
		virtual void print(const char * prefix) {
			for (auto c : computes) {
				FMT_ROOT_PRINT("{} MCP Method: {}\n", prefix, 
					(c == MCP2 ? "MCP2" : 
					(c == MCP3 ? "MCP3" : 
					(c == MCP4 ? "MCP4" : 
					(c == MU_MCP ? "Ensemble MCP" : 
					(c == MU_MCP_STOUFFER ? "Ensemble MCP with Stouffer" : 
					"unsupported")))))); 
			}
			FMT_ROOT_PRINT("{} MI method: {}\n", prefix, 
				(mi_computes == BSpline ? "bspline" : 
				(mi_computes == AP ? "adaptive partitioning with ranking" : "unknown"))); 
            FMT_ROOT_PRINT("{} MI file: {}\n", prefix, mi_file.c_str()); 
            FMT_ROOT_PRINT("{} coefficient input: {}\n", prefix, coeffs.c_str()); 
			FMT_ROOT_PRINT("{} MCP compute clamping output: {}\n", prefix, (clamped ? "true" : "false")); 
            FMT_ROOT_PRINT("{} groundtruth-list file: {}\n", prefix, groundtruth_list.c_str()); 
            FMT_ROOT_PRINT("{} groundtruth-matrix file: {}\n", prefix, groundtruth_mat.c_str()); 

            FMT_ROOT_PRINT("{} TF input: {}\n", prefix, tf_input.c_str()); 

			FMT_ROOT_PRINT("{} MI diagonal set to: {}\n", prefix, diagonal); 

			FMT_ROOT_PRINT("{} permute groundtruth to optimize combo: {}\n", prefix, (combo_from_train? "true" : "false")); 
			FMT_ROOT_PRINT("{} evaluation iterations set to: {}\n", prefix, iters); 

		}
        
};


// upper triangle only (should be same as lower triangle)
std::vector<std::tuple<size_t, size_t, int>> 
select_tfs_from_tf_mat(
	size_t const & local_rows, std::vector<double> const & tfs, 
	splash::ds::aligned_matrix<char> const & full_mat) {

	auto stime = getSysTime();
	splash::utils::partition<size_t> full_mat_part = splash::utils::partition<size_t>::make_partition(local_rows);

	std::vector<std::tuple<size_t, size_t, int>> mask;
	mask.reserve((local_rows * full_mat.columns()) >> 1);

	// scan and get local + elements.

	char v;
	const char * vptr;
	size_t rid = full_mat_part.offset;
	size_t last = full_mat.columns();
	for (size_t i = 0; i < full_mat_part.size; ++i, ++rid) {
		if (tfs[rid] < 0) continue;  // not a TF, so skip the row.
		vptr = full_mat.data(rid);
		for (size_t j = 0; j < last; ++j, ++vptr) {
			if (rid == j) continue;  // skip diagonal (on full matrix.)
			if ((rid > j) && (tfs[j] > 0))  continue;  // lower triangle AND target is a TF, so already included, don't add again.

			// if here, it's either the first TF1->TF2 encounter, or TF-TARGET.
			v = *vptr;
			if ((v == 1) || (v == 0)) {  
				mask.emplace_back(i, j, static_cast<int>(v));
			}
		}
	}

	auto etime = getSysTime();
	FMT_ROOT_PRINT("Computed Mask form mat in {} sec\n", get_duration_s(stime, etime));
	return mask;
}




// find difference in truth table. this is for making the test set
void subtract_truth(std::vector<std::tuple<size_t, size_t, int>> const & mask,
	std::vector<std::tuple<size_t, size_t, int>> const & part,
	std::vector<std::tuple<size_t, size_t, int>> & remaining) {
	
	auto comp = [](std::tuple<size_t, size_t, int> const & a, std::tuple<size_t, size_t, int> const & b){
		return (std::get<0>(a) == std::get<0>(b)) ? (std::get<1>(a) < std::get<1>(b)) : (std::get<0>(a) < std::get<0>(b));
	};

	// sort part
	std::vector<std::tuple<size_t, size_t, int>> sorted(part.begin(), part.end());
	std::sort(sorted.begin(), sorted.end(), comp);

	remaining.clear();
	remaining.reserve(mask.size() - part.size());
	// check and put into remaining.
	for (auto m : mask) {
		if (!std::binary_search(sorted.begin(), sorted.end(), m, comp)) {
			remaining.emplace_back(m);
		}
	}
}


// shuffle, then we can reuse multiple times.
void shuffle_truth(std::vector<std::tuple<size_t, size_t, int>> const & mask,
	int const & seed, 
	std::vector<std::tuple<size_t, size_t, int>> & shuffled) {

	// init random number generator
	// std::random_device rd;
	// std::mt19937 g(rd());
	std::mt19937 g(seed);

	// copy the mask
	shuffled.assign(mask.begin(), mask.end());

	// permute:
	std::shuffle(shuffled.begin(), shuffled.end(), g);
}

// compute the aupr for mcp (non-combo)
template <typename Kernel, typename T = double>
void eval_aupr_mcp(std::vector<std::tuple<size_t, size_t, int>> const & mask,
	// std::vector<std::tuple<size_t, size_t, int>> const & tf_mask,
	splash::ds::aligned_matrix<T> const & vals, 
	Kernel const & auprkern, size_t const & iterations, 
	splash::ds::aligned_vector<double> & pos, splash::ds::aligned_vector<double> & neg,
	bool const & tfonly, std::string const & tag) {

	// full groundtruth AUPR
	double aupr = compute_aupr(mask, vals, auprkern, pos, neg);
	FMT_ROOT_PRINT("AUPR {},{},,,sec\n", tag, aupr);

	// // TF vs non-TF AUPR
	// if (tf_mask.size() > 0) {
	// 	aupr = compute_aupr(tf_mask, vals, auprkern, pos, neg);
	// 	FMT_ROOT_PRINT("AUPR TF,{},,,sec\n", aupr);

	// 	// permuted set of training (ground truth)
	// 	std::vector<std::tuple<size_t, size_t, int>> non_tf_mask;
	// 	subtract_truth(mask, tf_mask, non_tf_mask);
	// 	aupr = compute_aupr(non_tf_mask, vals, auprkern, pos, neg);
	// 	FMT_ROOT_PRINT("AUPR Not_TF,{},,,sec\n", aupr);

	// }
	size_t masksize = mask.size();

	// get the counts for each count 
	std::vector<size_t> train_counts;
	train_counts.emplace_back(static_cast<double>(0.005) * static_cast<double>(masksize));
	train_counts.emplace_back(static_cast<double>(0.01) * static_cast<double>(masksize));
	train_counts.emplace_back(static_cast<double>(0.02) * static_cast<double>(masksize));
	train_counts.emplace_back(static_cast<double>(0.05) * static_cast<double>(masksize));
	train_counts.emplace_back(static_cast<double>(0.1) * static_cast<double>(masksize));
	if (tfonly) {
		train_counts.emplace_back(static_cast<double>(0.2) * static_cast<double>(masksize));
		train_counts.emplace_back(static_cast<double>(0.3) * static_cast<double>(masksize));
		train_counts.emplace_back(static_cast<double>(0.4) * static_cast<double>(masksize));
		train_counts.emplace_back(static_cast<double>(0.5) * static_cast<double>(masksize));
	}
	size_t parts = train_counts.size();

	std::vector<std::tuple<size_t, size_t, int>> shuffled;

	// compute the summaries.  4 in a row:  min, max, sum, sumsq
	std::vector<double> train_auprs;   // linearized 2D array, row major.  rows = a shuffle iteration, col = a decile.
	train_auprs.resize(4 * parts);
	std::vector<double> test_auprs;   // linearized 2D array, row major.  rows = a shuffle iteration, col = a decile.
	test_auprs.resize(4 * parts);
	for (size_t j = 0; j < parts; ++j) { 
		train_auprs[4 * j] = std::numeric_limits<double>::max(); 
		test_auprs[4 * j] = std::numeric_limits<double>::max(); 
	}

	for (size_t i = 1; i <= iterations; ++i) {
		// shuffle the groundtruth
		shuffle_truth(mask, i, shuffled);

		for (size_t j = 0; j < parts; ++j) {
			aupr = compute_aupr2(shuffled.begin(), shuffled.begin() + train_counts[j], vals, auprkern, pos, neg);
			train_auprs[(j << 2)] = std::min(train_auprs[(j << 2)], aupr);
			train_auprs[(j << 2) + 1] = std::max(train_auprs[(j << 2) + 1], aupr);
			train_auprs[(j << 2) + 2] += aupr;
			train_auprs[(j << 2) + 3] += aupr * aupr;
		 	
			aupr = compute_aupr2(shuffled.begin() + train_counts[j], shuffled.end(), vals, auprkern, pos, neg);
			test_auprs[(j << 2)] = std::min(test_auprs[(j << 2)], aupr);
			test_auprs[(j << 2) + 1] = std::max(test_auprs[(j << 2) + 1], aupr);
			test_auprs[(j << 2) + 2] += aupr;
			test_auprs[(j << 2) + 3] += aupr * aupr;
		}
	}

	// now compute min/max/mean/stdev for each proportion.
	for (size_t j = 0; j < parts; ++j) {
		FMT_ROOT_PRINT("AUPR {} train min,{},decile {},iters {},sec\n", tag, train_auprs[(j << 2)], j, iterations);
		FMT_ROOT_PRINT("AUPR {} train max,{},decile {},iters {},sec\n", tag, train_auprs[(j << 2) + 1], j, iterations);
		FMT_ROOT_PRINT("AUPR {} train mean,{},decile {},iters {},sec\n", tag, train_auprs[(j << 2) + 2] / static_cast<double>(iterations - 1), j, iterations);
		double mean = train_auprs[(j << 2) + 2] / static_cast<double>(iterations);
		double stdev = sqrt(static_cast<double>(iterations) / static_cast<double>(iterations-1) * (train_auprs[(j << 2) + 3] / static_cast<double>(iterations) - mean * mean) );
		FMT_ROOT_PRINT("AUPR {} train stdev,{},decile {},iters {},sec\n", tag, stdev, j, iterations);

		FMT_ROOT_PRINT("AUPR {} test min,{},decile {},iters {},sec\n", tag, test_auprs[(j << 2)], j, iterations);
		FMT_ROOT_PRINT("AUPR {} test max,{},decile {},iters {},sec\n", tag, test_auprs[(j << 2) + 1], j, iterations);
		FMT_ROOT_PRINT("AUPR {} test mean,{},decile {},iters {},sec\n", tag, test_auprs[(j << 2) + 2] / static_cast<double>(iterations - 1), j, iterations);
		mean = test_auprs[(j << 2) + 2] / static_cast<double>(iterations);
		stdev = sqrt(static_cast<double>(iterations) / static_cast<double>(iterations-1) * (test_auprs[(j << 2) + 3] / static_cast<double>(iterations) - mean * mean) );
		FMT_ROOT_PRINT("AUPR {} test stdev,{},decile {},iters {},sec\n", tag, stdev, j, iterations);
	}
}

template <typename MatrixType>
std::tuple<double, double, double, double, double, double> eval_combos(
	MatrixType const & coeffs, MatrixType const & mi, MatrixType dmaxmin1, 
	MatrixType const & dmaxmin2, MatrixType dmaxmin3, 
	std::vector<std::tuple<size_t, size_t, int>> const & mask, 
	std::vector<std::tuple<size_t, size_t, int>> const & mask_in, 
	std::vector<std::tuple<size_t, size_t, int>> const & mask_inv, 
	mcp::kernel::aupr_kernel<double, char, double> const & auprkern, 
	splash::ds::aligned_vector<double> & pos, 
	splash::ds::aligned_vector<double> & neg,
	bool const & tfonly, std::string const & tag,
	bool const & with_stouffer = true
	 
) {

	// ----------- compute the combos
	auto stime_all = getSysTime();
	auto stime = stime_all;
	auto etime = stime_all;
	MatrixType dratio, dratiost;
	double const * coeff = nullptr;

	// parameters for eval
	double max_aupr = 0, aupr = 0, aupr_in = 0, aupr_inv = 0;
	double max_saupr = 0, saupr = 0, saupr_in = 0, saupr_inv = 0;
	size_t co = 0, sco = 0;

	bool eval = true;
	
	for (size_t i = 0; i < coeffs.rows(); ++i) {
		coeff = coeffs.data(i);

		if ((coeff[0] == 1) && (coeff[1] == 0) && (coeff[2] == 0) && (coeff[3] == 0)) continue;

		// ------ combo ratio --------
		stime = getSysTime();
		compute_combo(mi, dmaxmin1, dmaxmin2, dmaxmin3, coeff, dratio);
		etime = getSysTime();
		FMT_ROOT_PRINT("Computed,Combo {},({}_{}_{}_{}),{},sec\n", 
			tag, coeff[0], coeff[1], coeff[2], coeff[3], get_duration_s(stime, etime));

		if (eval) {
			stime = getSysTime();
			aupr_in = compute_aupr(mask_in, dratio, auprkern, pos, neg);

			// save coeff if aupr is max.
			if (aupr_in > max_aupr) {
				max_aupr = aupr_in;
				co = i;
			}
			etime = getSysTime();
			FMT_ROOT_PRINT("Computed,Combo {} AUPR {},({}_{}_{}_{}),{},sec\n", 
				tag, aupr_in, coeff[0], coeff[1], coeff[2], coeff[3], get_duration_s(stime, etime));
		}

		// ------- stouffer --------
		if (with_stouffer) {
			stime = getSysTime();
			if (tfonly) compute_stouffer_tf(dratio, dratiost);  
			else compute_stouffer(dratio, dratiost); 
			// this is not correct.
			// compute_combo_params(gauss0, gauss1, gauss2, gauss3, coeff, gausses);
			// compute_stouffer2(dratio, gausses, dratiost);
			etime = getSysTime();
			FMT_ROOT_PRINT("Computed,Combo {} Stouffer,({}_{}_{}_{}),{},sec\n", 
				tag, coeff[0], coeff[1], coeff[2], coeff[3], get_duration_s(stime, etime));

			// write out again.
			if (eval) {
				
			//	for (size_t i = 0; i < dratiost.columns(); ++i) {
			//		FMT_ROOT_PRINT("{}, ", dratiost.at(0, i) );
			//	}

				stime = getSysTime();
				saupr_in = compute_aupr(mask_in, dratiost, auprkern, pos, neg);


				// save coeff if aupr is max.
				if (saupr_in > max_saupr) {
					max_saupr = saupr_in;
					sco = i;
				}
				etime = getSysTime();
				FMT_ROOT_PRINT("Computed,Combo {} Stouffer AUPR {},({}_{}_{}_{}),{},sec\n", 
					tag, saupr_in, coeff[0], coeff[1], coeff[2], coeff[3], get_duration_s(stime, etime));
			}
		}
	}

	if (eval) {
		// ------ combo ratio --------
		stime = getSysTime();
		compute_combo(mi, dmaxmin1, dmaxmin2, dmaxmin3, coeffs.data(co), dratio);
		etime = getSysTime();
		FMT_ROOT_PRINT("Computed,MAX Combo {} aupr_train {},({}_{}_{}_{}),{},sec\n", 
			tag, max_aupr, coeffs.at(co, 0), coeffs.at(co, 1), coeffs.at(co, 2), coeffs.at(co, 3), get_duration_s(stime, etime));
		stime = getSysTime();
		// aupr_in = compute_aupr(mask_in, dratio, auprkern, pos, neg);
		aupr_inv = compute_aupr(mask_inv, dratio, auprkern, pos, neg);
		etime = getSysTime();
		FMT_ROOT_PRINT("Computed,MAX Combo {} aupr_test {},,{},sec\n", tag, aupr_inv, get_duration_s(stime, etime));

		stime = getSysTime();
		// aupr_in = compute_aupr(mask_in, dratio, auprkern, pos, neg);
		aupr = compute_aupr(mask, dratio, auprkern, pos, neg);
		etime = getSysTime();
		FMT_ROOT_PRINT("Computed,MAX Combo {} aupr_full {},,{},sec\n", tag, aupr, get_duration_s(stime, etime));


		if (with_stouffer) {
			// ------- stouffer --------
			stime = getSysTime();
			compute_combo(mi, dmaxmin1, dmaxmin2, dmaxmin3, coeffs.data(sco), dratio);
			if (tfonly) compute_stouffer_tf(dratio, dratiost);  
			else compute_stouffer(dratio, dratiost); 
			etime = getSysTime();
			FMT_ROOT_PRINT("Computed,MAX Combo AND Stouffer {} aupr_train {},({}_{}_{}_{}),{},sec\n", 
				tag, max_saupr, coeffs.at(sco, 0), coeffs.at(sco, 1), coeffs.at(sco, 2), coeffs.at(sco, 3), get_duration_s(stime, etime));

			stime = getSysTime();
			// saupr_in = compute_aupr(mask_in, dratiost, auprkern, pos, neg);
			saupr_inv = compute_aupr(mask_inv, dratiost, auprkern, pos, neg);
			etime = getSysTime();
			FMT_ROOT_PRINT("Computed,MAX Combo AND Stouffer {} aupr_test {},,{},sec\n", tag, saupr_inv, get_duration_s(stime, etime));

			stime = getSysTime();
			// saupr_in = compute_aupr(mask_in, dratiost, auprkern, pos, neg);
			saupr = compute_aupr(mask, dratiost, auprkern, pos, neg);
			etime = getSysTime();
			FMT_ROOT_PRINT("Computed,MAX Combo AND Stouffer {} aupr_full {},,{},sec\n", tag, saupr, get_duration_s(stime, etime));
		}
	}

	auto etime_all = getSysTime();
	FMT_ROOT_PRINT("ALL Combos {} in {} sec\n", tag, get_duration_s(stime_all, etime_all));
	FMT_FLUSH();

	return std::make_tuple(max_aupr, aupr_inv, max_saupr, saupr_inv, aupr, saupr);

}

int main(int argc, char* argv[]) {

	//==============  PARSE INPUT =====================
	CLI::App app{"MI Combo Net"};

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

	size_t iterations = app_params.iters;


	// =============== SETUP INPUT ===================
	// NOTE: input data is replicated on all MPI procs.
	using MatrixType = splash::ds::aligned_matrix<double>;
	// using VectorType = splash::ds::aligned_vector<std::pair<double, double>>;
	MatrixType input;
	MatrixType mi;
	std::vector<std::string> genes;
	std::vector<std::string> samples;
	
	auto stime = getSysTime();
	auto etime = getSysTime();

	if (app_params.mi_file.length() == 0) {
		stime = getSysTime();
		input = read_matrix<double>(common_params.input, "array", 
				common_params.num_vectors, common_params.vector_size,
				genes, samples, common_params.skip );
		etime = getSysTime();
		FMT_ROOT_PRINT("Loaded,EXP,,{},sec\n", get_duration_s(stime, etime));
	} else {
		stime = getSysTime();
		mi = read_matrix<double>(app_params.mi_file, "array", 
				common_params.num_vectors, common_params.vector_size,
				genes, samples);
		etime = getSysTime();
		FMT_ROOT_PRINT("Loaded,EXP,,{},sec\n", get_duration_s(stime, etime));
	}


	bool tfonly = app_params.tf_input.length() > 0;
	// load the tf gene names and max values
	std::vector<double> tfs;
	std::vector<std::string> tf_names;
	if (tfonly) {
		stime = getSysTime();
		load_tfs(app_params.tf_input, genes, tfs, tf_names, mpi_params.rank);
		if ((app_params.tf_input.length() > 0) && (tf_names.size() == 0)) {
			FMT_ROOT_PRINT("ERROR: no transcription factors in the specified file");
			return 1;
		}
		etime = getSysTime();
		FMT_ROOT_PRINT("Loaded,TF,,{},sec\n", get_duration_s(stime, etime));
	}

	stime = getSysTime();
	bool eval_by_list = app_params.groundtruth_list.length() > 0;
	bool eval_by_mat = app_params.groundtruth_mat.length() > 0;
	std::vector<std::tuple<std::string, std::string, int>> truth, tf_truth;
	splash::ds::aligned_matrix<char> truth_mat, tf_truth_mat;
	splash::ds::aligned_vector<double> pos;
	splash::ds::aligned_vector<double> neg;
	std::vector<std::tuple<size_t, size_t, int>> mask, tf_mask;
	mcp::kernel::aupr_kernel<double, char, double> auprkern;
	mcp::kernel::auroc_kernel<double, char, double> aurockern;

	size_t rows = common_params.num_vectors, columns = common_params.vector_size;
	std::vector<std::string> genes2;
	std::vector<std::string> samples2;



	if (eval_by_list) {
		truth = read_groundtruth(app_params.groundtruth_list);
	} else if (eval_by_mat) {
		truth_mat = read_matrix<char>(app_params.groundtruth_mat, "array", 
			rows, columns,
			genes2, samples2);
	}

	if (tfonly) {
		if (eval_by_list) {
			// and filter
			filter_list_by_tfs(truth, tf_names, tf_truth);
		} else if (eval_by_mat) {
			filter_mat_rows_by_tfs(truth_mat, tfs, tf_truth_mat);
		}
	}

	etime = getSysTime();
	FMT_ROOT_PRINT("Loaded,truth,,{},sec\n", get_duration_s(stime, etime));


	if (mpi_params.rank == 0) {
		mpi_params.print("[PARAM] ");
		common_params.print("[PARAM] ");
		app_params.print("[PARAM] ");
	}

	std::string output_prefix = common_params.output;
	if (tfonly) {
		output_prefix.append("_tf");
	}

	if (app_params.mi_computes == app_parameters::mi_method_type::BSpline)
		output_prefix.append("_bs");
	else 
		output_prefix.append("_ap");



	// =============== MI ===================
	MatrixType dmi;
	if (app_params.mi_file.length() == 0) {
		dmi.resize(input.rows(), input.columns());
		stime = getSysTime();
		if (app_params.mi_computes == app_parameters::mi_method_type::BSpline)
			compute_mi(input, app_params.num_bins, dmi);
		else  
			compute_ap_mi(input, dmi, true);  // add noise.
		mi = dmi.allgather();
		etime = getSysTime();
		FMT_ROOT_PRINT("Computed,MI,,{},sec\n", get_duration_s(stime, etime));
	}

	// set diagonal
	if (app_params.diagonal >= 0.0) {
		stime = getSysTime();
		size_t mc = std::min(mi.rows(), mi.columns());
		for (size_t i = 0; i < mc; ++i) {
			mi.at(i, i) = app_params.diagonal;
		}
		etime = getSysTime();
		FMT_ROOT_PRINT("Set MI diagonal {},,{},sec\n", app_params.diagonal, get_duration_s(stime, etime));
	}
	// distribute mi to dmi.
	stime = getSysTime();
	dmi = mi.scatter(0);
	etime = getSysTime();
	FMT_ROOT_PRINT("Scatter MI,,{},sec\n", get_duration_s(stime, etime));
	
	// if TF, then need to do some filtering for later use.
	MatrixType dmi_tf, mi_tf;
	if (tfonly) {
		stime = getSysTime();
		filter_local_mat_rows_by_tfs(dmi, tfs, dmi_tf);
		mi_tf = dmi_tf.allgather();
		// rebalance
		dmi_tf = mi_tf.scatter(0);
		etime = getSysTime();
		FMT_ROOT_PRINT("Get TF MI,,{},sec\n", get_duration_s(stime, etime));
	}
	

	if (eval_by_list) {

		stime = getSysTime();
		if (tfonly) tf_mask = select_from_list(dmi_tf.rows(), tf_truth, tf_names, genes);
		else mask = select_from_list(dmi.rows(), truth, genes, genes);

		etime = getSysTime();
		FMT_ROOT_PRINT("Computed,truth mask,,{},sec\n", get_duration_s(stime, etime));

	} else if (eval_by_mat) {
		stime = getSysTime();
		
		if (tfonly) tf_mask = select_tfs_from_mat(dmi_tf.rows(), tfs, tf_truth_mat);
		else mask = select_lower_triangle(dmi.rows(), truth_mat);
		etime = getSysTime();
		FMT_ROOT_PRINT("Computed,truth mask mat,,{},sec\n", get_duration_s(stime, etime));
	}

	if (eval_by_list || eval_by_mat) {
		
		if (tfonly) eval_aupr_mcp(tf_mask, dmi_tf, auprkern, iterations, pos, neg, tfonly, "TF");
		else eval_aupr_mcp(mask, dmi, auprkern, iterations, pos, neg, tfonly, "full");
	}

	// =============== MCP 2, 3, 4 and ensemble ========
	std::unordered_set<app_parameters::method_type> comps;
	for (auto c : app_params.computes) {
		comps.insert(c);
	}



	MatrixType dratio;
	if (comps.find(app_parameters::method_type::MCP2) != comps.end()) {
		// ------ ratio 1 --------
		stime = getSysTime();
		if (tfonly)
			compute_mcp2_tfs(mi_tf, mi, 1, tfs, dratio);
		else 
			compute_mcp2(mi, dratio);
		etime = getSysTime();
		FMT_ROOT_PRINT("Computed,MCP2,,{},sec\n", get_duration_s(stime, etime));


	if (eval_by_list) {

		stime = getSysTime();
		
		if (tfonly) tf_mask = select_from_list(dratio.rows(), tf_truth, tf_names, genes);
		else mask = select_from_list(dratio.rows(), truth, genes, genes);
		etime = getSysTime();
		FMT_ROOT_PRINT("Computed,truth mask,,{},sec\n", get_duration_s(stime, etime));

	} else if (eval_by_mat) {
		stime = getSysTime();
		if (tfonly) tf_mask = select_tfs_from_mat(dratio.rows(), tfs, tf_truth_mat);
		else 		mask = select_lower_triangle(dratio.rows(), truth_mat);

		etime = getSysTime();
		FMT_ROOT_PRINT("Computed,truth mask mat,,{},sec\n", get_duration_s(stime, etime));
	}



		if (eval_by_list || eval_by_mat) {
			if (tfonly) eval_aupr_mcp(tf_mask, dratio, auprkern, iterations, pos, neg, tfonly, "TF");
			else eval_aupr_mcp(mask, dratio, auprkern, iterations, pos, neg, tfonly, "full");
		}
		FMT_FLUSH();
	} 
	
	dratio.zero();
	if (comps.find(app_parameters::method_type::MCP3) != comps.end()) {
		// ------ ratio 1 --------
		stime = getSysTime();
		if (tfonly) compute_mcp3_tfs(mi_tf, mi, 2, tfs, dratio);
		else compute_mcp3(mi, dratio);
		etime = getSysTime();
		FMT_ROOT_PRINT("Computed,MCP3,,{},sec\n", get_duration_s(stime, etime));
		
	if (eval_by_list) {

		stime = getSysTime();
		if (tfonly) tf_mask = select_from_list(dratio.rows(), tf_truth, tf_names, genes);
		else mask = select_from_list(dratio.rows(), truth, genes, genes);
		etime = getSysTime();
		FMT_ROOT_PRINT("Computed,truth mask,,{},sec\n", get_duration_s(stime, etime));

		// stime = getSysTime();
		// mask2 = select_mask2(truth, dsrc_names, dst_names);
		// etime = getSysTime();
		// FMT_ROOT_PRINT("Computed,truth mask2,,{},sec\n", get_duration_s(stime, etime));
		// end onetime.
	} else if (eval_by_mat) {
		stime = getSysTime();
		if (tfonly) tf_mask = select_tfs_from_mat(dratio.rows(), tfs, tf_truth_mat);
		else 		mask = select_lower_triangle(dratio.rows(), truth_mat);
		etime = getSysTime();
		FMT_ROOT_PRINT("Computed,truth mask mat,,{},sec\n", get_duration_s(stime, etime));
	}

		if (eval_by_list || eval_by_mat) {
			if (tfonly) eval_aupr_mcp(tf_mask, dratio, auprkern, iterations, pos, neg, tfonly, "TF");
			else eval_aupr_mcp(mask, dratio, auprkern, iterations, pos, neg, tfonly, "full");
		}

		FMT_FLUSH();
	} 

	dratio.zero();
	if (comps.find(app_parameters::method_type::MCP4) != comps.end()) {
		// ------ ratio 1 --------
		stime = getSysTime();
		if (tfonly)
			compute_mcp4_tfs(mi_tf, mi, 3, tfs, dratio);
		else 
			compute_mcp4(mi, dratio);
		etime = getSysTime();
		FMT_ROOT_PRINT("Computed,MCP4,,{},sec\n", get_duration_s(stime, etime));

	if (eval_by_list) {

		stime = getSysTime();
		if (tfonly) tf_mask = select_from_list(dratio.rows(), tf_truth, tf_names, genes);
		else mask = select_from_list(dratio.rows(), truth, genes, genes);
		etime = getSysTime();
		FMT_ROOT_PRINT("Computed,truth mask,,{},sec\n", get_duration_s(stime, etime));

	} else if (eval_by_mat) {
		stime = getSysTime();
		if (tfonly) tf_mask = select_tfs_from_mat(dratio.rows(), tfs, tf_truth_mat);
		else 		mask = select_lower_triangle(dratio.rows(), truth_mat);
		etime = getSysTime();
		FMT_ROOT_PRINT("Computed,truth mask mat,,{},sec\n", get_duration_s(stime, etime));
	}

		if (eval_by_list || eval_by_mat) {
			if (tfonly) eval_aupr_mcp(tf_mask, dratio, auprkern, iterations, pos, neg, tfonly, "TF");
			else eval_aupr_mcp(mask, dratio, auprkern, iterations, pos, neg, tfonly, "full");
		}
		FMT_FLUSH();
	} 
	
	dratio.zero();
	bool with_stouffer = (comps.find(app_parameters::method_type::MU_MCP_STOUFFER) != comps.end());
	// ============= compute combos ====================
	if ((comps.find(app_parameters::method_type::MU_MCP) != comps.end()) || 
	  with_stouffer) {
		// ----------- read the coefficient file.
		stime = getSysTime();
		MatrixType coeffs;
		size_t rs = 0;
		size_t cs = 0;
		{
			if (app_params.coeffs.length() > 0) {
				std::vector<std::string> rdummy;
				std::vector<std::string> cdummy;
				coeffs = read_matrix<double>(app_params.coeffs, "array", 
					rs, cs,
					rdummy, cdummy);
			} else {
				rs = 1;
				cs = 4;
				coeffs.resize(1, 4);
				coeffs.at(0, 0) = 0.25;
				coeffs.at(0, 1) = 0.25;
				coeffs.at(0, 2) = 0.25;
				coeffs.at(0, 3) = 0.25;
			}
		}
		etime = getSysTime();
		FMT_ROOT_PRINT("Loaded,{}x{} coefficients,,{},sec\n", rs, cs, coeffs.rows(), coeffs.columns(), get_duration_s(stime, etime));

		// ----------- get all intermediate results
		stime = getSysTime();
		MatrixType dmaxmin1, dmaxmin2, dmaxmin3;
		if (tfonly)
			compute_maxmins_tfs(mi_tf, mi, tfs, dmaxmin1, dmaxmin2, dmaxmin3);
		else
			compute_maxmins(mi, dmaxmin1, dmaxmin2, dmaxmin3);

		etime = getSysTime();
		FMT_ROOT_PRINT("Computed,MAXMIN 1 2 3,,{},sec\n", get_duration_s(stime, etime));


	if (eval_by_list) {

		stime = getSysTime();
		if (tfonly) tf_mask = select_from_list(dmaxmin1.rows(), tf_truth, tf_names, genes);
		else mask = select_from_list(dmaxmin1.rows(), truth, genes, genes);
		etime = getSysTime();
		FMT_ROOT_PRINT("Computed,truth mask,,{},sec\n", get_duration_s(stime, etime));

	} else if (eval_by_mat) {
		stime = getSysTime();
		if (tfonly) tf_mask = select_tfs_from_mat(dmaxmin1.rows(), tfs, tf_truth_mat);
		else 		mask = select_lower_triangle(dmaxmin1.rows(), truth_mat);
		etime = getSysTime();
		FMT_ROOT_PRINT("Computed,truth mask mat,,{},sec\n", get_duration_s(stime, etime));
	}

		FMT_ROOT_PRINT("Computing,percent 100,iters ,sec\n");

		// whole dataset.
		std::vector<std::tuple<size_t, size_t, int>> mask_inv;
		std::tuple<double, double, double, double, double, double> auprs; 
		std::string tag = (tfonly ? "TF" : "full");
		size_t masksize = (tfonly ? tf_mask.size() : mask.size());

		if (app_params.combo_from_train) {
		// looking at 10% ground truth
		std::vector<size_t> train_counts;
		train_counts.emplace_back(static_cast<double>(0.005) * static_cast<double>(masksize));
		train_counts.emplace_back(static_cast<double>(0.01) * static_cast<double>(masksize));
		train_counts.emplace_back(static_cast<double>(0.02) * static_cast<double>(masksize));
		train_counts.emplace_back(static_cast<double>(0.05) * static_cast<double>(masksize));
		train_counts.emplace_back(static_cast<double>(0.1) * static_cast<double>(masksize));
		if (tfonly) {
			train_counts.emplace_back(static_cast<double>(0.2) * static_cast<double>(masksize));
			train_counts.emplace_back(static_cast<double>(0.3) * static_cast<double>(masksize));
			train_counts.emplace_back(static_cast<double>(0.4) * static_cast<double>(masksize));
			train_counts.emplace_back(static_cast<double>(0.5) * static_cast<double>(masksize));
		}
		size_t parts = train_counts.size();

		std::vector<std::tuple<size_t, size_t, int>> shuffled;

		std::vector<double> train_auprs;   // linearized 2D array, row major.  rows = a shuffle iteration, col = a decile.
		train_auprs.resize(4 * parts);
		std::vector<double> test_auprs;   // linearized 2D array, row major.  rows = a shuffle iteration, col = a decile.
		test_auprs.resize(4 * parts);
		std::vector<double> train_sauprs;   // linearized 2D array, row major.  rows = a shuffle iteration, col = a decile.
		train_sauprs.resize(4 * parts);
		std::vector<double> test_sauprs;   // linearized 2D array, row major.  rows = a shuffle iteration, col = a decile.
		test_sauprs.resize(4 * parts);
		std::vector<double> full_auprs;   // linearized 2D array, row major.  rows = a shuffle iteration, col = a decile.
		full_auprs.resize(4 * parts);
		std::vector<double> full_sauprs;   // linearized 2D array, row major.  rows = a shuffle iteration, col = a decile.
		full_sauprs.resize(4 * parts);
		for (size_t j = 0; j < parts; ++j) { 
			train_auprs[4 * j] = std::numeric_limits<double>::max(); 
			test_auprs[4 * j] = std::numeric_limits<double>::max(); 
			full_auprs[4 * j] = std::numeric_limits<double>::max(); 
			train_sauprs[4 * j] = std::numeric_limits<double>::max(); 
			test_sauprs[4 * j] = std::numeric_limits<double>::max(); 
			full_sauprs[4 * j] = std::numeric_limits<double>::max(); 
		}
		std::vector<std::tuple<size_t, size_t, int>> mask_in;

		for (size_t i = 1; i <= iterations; ++i) {
			// shuffle the groundtruth
			if (tfonly) shuffle_truth(tf_mask, i, shuffled);
			else shuffle_truth(mask, i, shuffled);

			for (size_t j = 0; j < parts; ++j) {
				mask_in.assign(shuffled.begin(), shuffled.begin() + train_counts[j]);
				mask_inv.assign(shuffled.begin() + train_counts[j], shuffled.end());
				
				FMT_ROOT_PRINT("Computing, {} percent {},iters {},sec\n", tag, static_cast<double>(mask_in.size() * 100) / static_cast<double>(masksize) , i);
					
				auprs =
					eval_combos(coeffs, (tfonly ? mi_tf : mi), dmaxmin1, dmaxmin2, dmaxmin3, (tfonly ? tf_mask : mask), mask_in, mask_inv, auprkern, pos, neg, tfonly, tag, with_stouffer);

				auto aupr = std::get<0>(auprs);
				train_auprs[(j << 2) + 0] = std::min(train_auprs[(j << 2) + 0], aupr);
				train_auprs[(j << 2) + 1] = std::max(train_auprs[(j << 2) + 1], aupr);
				train_auprs[(j << 2) + 2] += aupr;
				train_auprs[(j << 2) + 3] += aupr * aupr;
				
				aupr = std::get<1>(auprs);
				test_auprs[(j << 2) + 0] = std::min(test_auprs[(j << 2) + 0], aupr);
				test_auprs[(j << 2) + 1] = std::max(test_auprs[(j << 2) + 1], aupr);
				test_auprs[(j << 2) + 2] += aupr;
				test_auprs[(j << 2) + 3] += aupr * aupr;

				aupr = std::get<2>(auprs);
				train_sauprs[(j << 2) + 0] = std::min(train_sauprs[(j << 2) + 0], aupr);
				train_sauprs[(j << 2) + 1] = std::max(train_sauprs[(j << 2) + 1], aupr);
				train_sauprs[(j << 2) + 2] += aupr;
				train_sauprs[(j << 2) + 3] += aupr * aupr;
				
				aupr = std::get<3>(auprs);
				test_sauprs[(j << 2) + 0] = std::min(test_sauprs[(j << 2) + 0], aupr);
				test_sauprs[(j << 2) + 1] = std::max(test_sauprs[(j << 2) + 1], aupr);
				test_sauprs[(j << 2) + 2] += aupr;
				test_sauprs[(j << 2) + 3] += aupr * aupr;

				aupr = std::get<4>(auprs);
				full_auprs[(j << 2) + 0] = std::min(full_auprs[(j << 2) + 0], aupr);
				full_auprs[(j << 2) + 1] = std::max(full_auprs[(j << 2) + 1], aupr);
				full_auprs[(j << 2) + 2] += aupr;
				full_auprs[(j << 2) + 3] += aupr * aupr;

				aupr = std::get<5>(auprs);
				full_sauprs[(j << 2) + 0] = std::min(full_sauprs[(j << 2) + 0], aupr);
				full_sauprs[(j << 2) + 1] = std::max(full_sauprs[(j << 2) + 1], aupr);
				full_sauprs[(j << 2) + 2] += aupr;
				full_sauprs[(j << 2) + 3] += aupr * aupr;

			}

			// output every 10 iterations, because this is very long running.
			if ((i % 10) == 0) {
				for (size_t j = 0; j < parts; ++j) {
					FMT_ROOT_PRINT("AUPR {} train min,{},decile {},iters {},sec\n", tag,train_auprs[(j << 2)], j, i);
					FMT_ROOT_PRINT("AUPR {} train max,{},decile {},iters {},sec\n", tag,train_auprs[(j << 2) + 1], j, i);
					FMT_ROOT_PRINT("AUPR {} train mean,{},decile {},iters {},sec\n",tag, train_auprs[(j << 2) + 2] / static_cast<double>(i - 1), j, i);
					double mean = train_auprs[(j << 2) + 2] / static_cast<double>(i);
					double stdev = sqrt(static_cast<double>(i) / static_cast<double>(i-1) * (train_auprs[(j << 2) + 3] / static_cast<double>(i) - mean * mean) );
					FMT_ROOT_PRINT("AUPR {} train stdev,{},decile {},iters {},sec\n", tag, stdev, j, i);

					FMT_ROOT_PRINT("AUPR {} test min,{},decile {},iters {},sec\n", tag, test_auprs[(j << 2)], j, i);
					FMT_ROOT_PRINT("AUPR {} test max,{},decile {},iters {},sec\n", tag, test_auprs[(j << 2) + 1], j, i);
					FMT_ROOT_PRINT("AUPR {} test mean,{},decile {},iters {},sec\n",tag,  test_auprs[(j << 2) + 2] / static_cast<double>(i - 1), j, i);
					mean = test_auprs[(j << 2) + 2] / static_cast<double>(i);
					stdev = sqrt(static_cast<double>(i) / static_cast<double>(i-1) * (test_auprs[(j << 2) + 3] / static_cast<double>(i) - mean * mean) );
					FMT_ROOT_PRINT("AUPR {} test stdev,{},decile {},iters {},sec\n", tag, stdev, j, i);

					FMT_ROOT_PRINT("AUPR stouffer {} train min,{},decile {},iters {},sec\n", tag, train_sauprs[(j << 2)], j, i);
					FMT_ROOT_PRINT("AUPR stouffer {} train max,{},decile {},iters {},sec\n", tag, train_sauprs[(j << 2) + 1], j, i);
					FMT_ROOT_PRINT("AUPR stouffer {} train mean,{},decile {},iters {},sec\n",tag,  train_sauprs[(j << 2) + 2] / static_cast<double>(i - 1), j, i);
					mean = train_sauprs[(j << 2) + 2] / static_cast<double>(i);
					stdev = sqrt(static_cast<double>(i) / static_cast<double>(i-1) * (train_sauprs[(j << 2) + 3] / static_cast<double>(i) - mean * mean) );
					FMT_ROOT_PRINT("AUPR stouffer {} train stdev,{},decile {},iters {},sec\n", tag, stdev, j, i);

					FMT_ROOT_PRINT("AUPR stouffer {} test min,{},decile {},iters {},sec\n", tag, test_sauprs[(j << 2)], j, i);
					FMT_ROOT_PRINT("AUPR stouffer {} test max,{},decile {},iters {},sec\n", tag, test_sauprs[(j << 2) + 1], j, i);
					FMT_ROOT_PRINT("AUPR stouffer {} test mean,{},decile {},iters {},sec\n",tag,  test_sauprs[(j << 2) + 2] / static_cast<double>(i - 1), j, i);
					mean = test_sauprs[(j << 2) + 2] / static_cast<double>(i);
					stdev = sqrt(static_cast<double>(i) / static_cast<double>(i-1) * (test_sauprs[(j << 2) + 3] / static_cast<double>(i) - mean * mean) );
					FMT_ROOT_PRINT("AUPR stouffer {} test stdev,{},decile {},iters {},sec\n", tag, stdev, j, i);

					FMT_ROOT_PRINT("AUPR {} full min,{},decile {},iters {},sec\n", tag, full_auprs[(j << 2)], j, i);
					FMT_ROOT_PRINT("AUPR {} full max,{},decile {},iters {},sec\n", tag, full_auprs[(j << 2) + 1], j, i);
					FMT_ROOT_PRINT("AUPR {} full mean,{},decile {},iters {},sec\n",tag,  full_auprs[(j << 2) + 2] / static_cast<double>(i - 1), j, i);
					mean = full_auprs[(j << 2) + 2] / static_cast<double>(i);
					stdev = sqrt(static_cast<double>(i) / static_cast<double>(i-1) * (full_auprs[(j << 2) + 3] / static_cast<double>(i) - mean * mean) );
					FMT_ROOT_PRINT("AUPR {} full stdev,{},decile {},iters {},sec\n", tag, stdev, j, i);

					FMT_ROOT_PRINT("AUPR stouffer {} full min,{},decile {},iters {},sec\n", tag, full_sauprs[(j << 2)], j, i);
					FMT_ROOT_PRINT("AUPR stouffer {} full max,{},decile {},iters {},sec\n", tag, full_sauprs[(j << 2) + 1], j, i);
					FMT_ROOT_PRINT("AUPR stouffer {} full mean,{},decile {},iters {},sec\n",tag,  full_sauprs[(j << 2) + 2] / static_cast<double>(i - 1), j, i);
					mean = full_sauprs[(j << 2) + 2] / static_cast<double>(i);
					stdev = sqrt(static_cast<double>(i) / static_cast<double>(i-1) * (full_sauprs[(j << 2) + 3] / static_cast<double>(i) - mean * mean) );
					FMT_ROOT_PRINT("AUPR stouffer {} full stdev,{},decile {},iters {},sec\n", tag, stdev, j, i);

				}

			}
		}
		}

	}

	return 0;
}
