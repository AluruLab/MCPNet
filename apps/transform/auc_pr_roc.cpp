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

#include "CLI/CLI.hpp"
#include "splash/io/CLIParserCommon.hpp"
#include "splash/io/parameters_base.hpp"
#include "splash/utils/benchmark.hpp"
#include "splash/utils/report.hpp"
#include "splash/ds/aligned_matrix.hpp"
#include "splash/patterns/pattern.hpp"

#include "mcp/transform/aupr.hpp"
#include "mcp/transform/auroc.hpp"

#include "splash/io/matrix_io.hpp"


#ifdef USE_OPENMP
#include <omp.h>
#endif


// combined, end to end, for generation and testing.
// this imple avoids file IO.

class app_parameters : public parameters_base {
	public:

		std::string groundtruth_list;
		std::string groundtruth_mat;

		app_parameters() {}
		virtual ~app_parameters() {}

		virtual void config(CLI::App& app) {
			auto gt_list_opt = app.add_option("-g,--groundtruth-list", groundtruth_list, "filename of groundtruth edge list.  require just forward or reverse to be present. (+) -> 1, (-) -> 0, unknown -> -1");
			auto gt_mat_opt = app.add_option("-x,--groundtruth-matrix", groundtruth_mat, "filename of groundtruth matrix. require symmetric matrix. (+) -> 1, (-) -> 0, unknown -> -1");

			auto opt_group = app.add_option_group("evaluation");
            opt_group->add_option(gt_list_opt);
            opt_group->add_option(gt_mat_opt);
            opt_group->require_option(1);

		}
		virtual void print(const char * prefix) {
            FMT_ROOT_PRINT("{} groundtruth-list file: {}\n", prefix, groundtruth_list.c_str()); 
            FMT_ROOT_PRINT("{} groundtruth-matrix file: {}\n", prefix, groundtruth_mat.c_str()); 
		}
        
};



// read and sort it, so rows grouped together.  not mapped to row order.
std::vector<std::tuple<std::string, std::string, int>> read_groundtruth(std::string const & truthfile) {

	std::vector<std::tuple<std::string, std::string, int>> content;  // need to store this to sort.

	auto stime = getSysTime();
	std::ifstream infile(truthfile);
	std::string s, t;
	int g;

	while (infile >> s >> t >> g)
	{
		content.emplace_back(s,t,g);
	}
	auto etime = getSysTime();
	FMT_ROOT_PRINT("read truth file in {} sec\n", get_duration_s(stime, etime));

	stime = getSysTime();
	std::sort(content.begin(), content.end(), 
		[](std::tuple<std::string, std::string, int> const & x, 
			std::tuple<std::string, std::string, int> const & y){
				auto val = std::get<0>(x).compare(std::get<0>(y));
			return (val == 0) ? (std::get<1>(x) < std::get<1>(y)) : (val < 0);
	} );
	etime = getSysTime();
	FMT_ROOT_PRINT("sort truth in {} sec\n", get_duration_s(stime, etime));

	return content;
}


// for search.  no longer correlate to row order.  NOT USED
void sort_rowcol_names(
	std::vector<std::string> const & source_gene,
	std::vector<std::string> const & target_gene,
	std::vector<std::pair<std::string, size_t>> & src_names,
	std::vector<std::pair<std::string, size_t>> & dst_names) {

	// generate sorted row and col names, with id to original array.
	auto comp = [](std::pair<std::string, size_t> const & x, std::pair<std::string, size_t> const & y) {
		return x.first < y.first;
	};

	src_names.clear();
	src_names.reserve(source_gene.size());
	for (size_t i = 0; i < source_gene.size(); ++i) {
		src_names.emplace_back(source_gene[i], i);
	}
	std::sort(src_names.begin(), src_names.end(), comp); 

	dst_names.clear();
	dst_names.reserve(target_gene.size());
	for (size_t i = 0; i < target_gene.size(); ++i) {
		dst_names.emplace_back(target_gene[i], i);
	}
	std::sort(dst_names.begin(), dst_names.end(), comp);

}

// for distributed matrix, where split is via rows.
// sort within each partition only.  for search only.
void sort_rowcol_names(
	size_t const & local_rows,
	std::vector<std::string> const & source_gene,
	std::vector<std::string> const & target_gene,
	std::vector<std::pair<std::string, size_t>> & dsrc_names,
	std::vector<std::pair<std::string, size_t>> & dst_names) {

	// generate sorted row and col names, with id to original array.
	auto comp = [](std::pair<std::string, size_t> const & x, std::pair<std::string, size_t> const & y) {
		return x.first < y.first;
	};

	splash::utils::partition<size_t> mpi_part = splash::utils::partition<size_t>::make_partition(local_rows);

	dsrc_names.clear();
	dsrc_names.reserve(mpi_part.size);
	size_t id = mpi_part.offset;
	for (size_t i = 0; i < mpi_part.size; ++i, ++id) {
		dsrc_names.emplace_back(source_gene[id], i);
	}
	std::sort(dsrc_names.begin(), dsrc_names.end(), comp); 

	dst_names.clear();
	dst_names.reserve(target_gene.size());
	for (size_t i = 0; i < target_gene.size(); ++i) {
		dst_names.emplace_back(target_gene[i], i);
	}
	std::sort(dst_names.begin(), dst_names.end(), comp);

}


// convert the truth list to a list of numeric coordinates, which can be applied repeatedly. truth is always ful.  src_name can be distributed, locally sorted, with local coords.
// l (|content|), m (|src|), n (|dest|)
// approaches:  
// 1.  iterate over truth and search in src and dst.   truth needs to be grouped,  src and dst need to be sorted.
//    l log m (search src, skip if same row) + l log n (search dest)
// 2.  iterate over src and search in truth.  truth and dest need to be sorted, src does not 
//    m log l ( search truth for row) + l log n (search dest)
// so decision is which is smaller, l or m.  note that if we do linear scan for dest, we could end up with nl.
// however, since we are doing binary search, larger one has more random memory access.  so let's use opposite decision.
std::vector<std::tuple<size_t, size_t, int>> 
select_mask(std::vector<std::tuple<std::string, std::string, int>> const & truth,
	std::vector<std::pair<std::string, size_t>> const & src_names,
	std::vector<std::pair<std::string, size_t>> const & dst_names) {

	auto stime = getSysTime();

	std::vector<std::tuple<size_t, size_t, int>> mask;
	mask.reserve(truth.size());

	std::string last_s, s, t;

	auto comp = [](std::pair<std::string, size_t> const & x, std::pair<std::string, size_t> const & y) {
		return x.first < y.first;
	};

	// all lists are sorted.  linear search should be faster for rows..
	auto it = truth.begin();
	auto s_it = src_names.begin();
	auto t_it = dst_names.begin();
	std::pair<std::string, size_t> query = {"", 0};

	// go through the truth table.
	if (truth.size() < src_names.size()) {
		// truth smaller.  do 2. avoid binary search with larger.  since ordered, avoid binary search.
		for (auto sn : src_names) {
			// this part ultimately goes over the array 
			while ((it != truth.end()) && (std::get<0>(*it) < sn.first)) ++it;  // find first s_it >= s

			// walk through truth
			t_it = dst_names.begin();   // reset
			for (; (it != truth.end()) && (std::get<0>(*it) == sn.first); ++it) {
				query.first = std::get<1>(*it);

				if (sn.first == query.first) continue;  // skip diagonals.

				t_it = std::lower_bound(t_it, dst_names.end(), query, comp);  // since both truth and dst_names are ordered for t,  can reduce search space each time.
				if ((t_it != dst_names.end()) && (t_it->first == query.first)) {
					// found one.
					mask.emplace_back(sn.second, t_it->second, std::get<2>(*it));
				} // else, nothing was found, so mask addition here. 

			} // at the end, or did not find == s
		}

	} else {
		last_s = "";
		s_it = src_names.begin();
		// truth larger.  do 1. iterate over truth
		for (; it != truth.end(); ++it) {
			s = std::get<0>(*it);
			t = std::get<1>(*it);

			if (s == t) continue;  // skip diagonals

			if (last_s != s) { // only if this is a new row.
				// search in src.
				query.first = s;
				s_it = std::lower_bound(s_it, src_names.end(), query, comp);  // since both truth and dst_names are ordered for t,  can reduce search space each time.
				t_it = dst_names.begin();  // reset to start of row
				last_s = s;
			}
			query.first = t;
			t_it = std::lower_bound(t_it, dst_names.end(), query, comp);  // since both truth and dst_names are ordered for t,  can reduce search space each time.
				
			if ((s_it != src_names.end()) && (t_it != dst_names.end()) &&
				(s_it->first == s) && (t_it->first == t)) {  // exact match
					// found one.
					mask.emplace_back(s_it->second, t_it->second, std::get<2>(*it));
			} // else, nothing was found, so mask addition here. 

		}
	}

	auto etime = getSysTime();
	FMT_ROOT_PRINT("Computed mask in {} sec\n", get_duration_s(stime, etime));

	stime = getSysTime();
	std::sort(mask.begin(), mask.end(), 
		[](std::tuple<size_t, size_t, int> const & x, std::tuple<size_t, size_t, int> const & y){
			return (std::get<0>(x) == std::get<0>(y)) ? 
				(std::get<1>(x) < std::get<1>(y)) : (std::get<0>(x) < std::get<0>(y));
		});
	etime = getSysTime();
	FMT_ROOT_PRINT("Sorted mask in {} sec\n", get_duration_s(stime, etime));

	return mask;
}

// upper triangle only (should be same as lower triangle)
std::vector<std::tuple<size_t, size_t, int>> 
select_lower_triangle(
	size_t const & local_rows,
	splash::ds::aligned_matrix<char> const & mat) {

	auto stime = getSysTime();
	splash::utils::partition<size_t> mpi_part = splash::utils::partition<size_t>::make_partition(local_rows);

	std::vector<std::tuple<size_t, size_t, int>> mask;
	mask.reserve((local_rows * mat.columns()) >> 1);

	char v;
	const char * vptr;
	size_t rid = mpi_part.offset;
	size_t last;
	for (size_t i = 0; i < mpi_part.size; ++i, ++rid) {
		vptr = mat.data(rid);
		last = std::min(mat.columns(), rid);
		for (size_t j = 0; j < last; ++j, ++vptr) {
			v = *vptr;
			if ((v == 1) || (v == 0)) {
				mask.emplace_back(rid, j, static_cast<int>(v));
			}
		}
	}

	auto etime = getSysTime();
	FMT_ROOT_PRINT("Computed Mask form mat in {} sec\n", get_duration_s(stime, etime));
	return mask;
}

// walking through appears to work properly.  negative probs all come before positive probes.
// aupr code is validated to work correctly
// mask:  sorted.  (once)  could be distributed.
// value: could be distribued.
void select_values(std::vector<std::tuple<size_t, size_t, int>> const & mask,
	splash::ds::aligned_matrix<double> const & values, 
	splash::ds::aligned_vector<double> & pos, splash::ds::aligned_vector<double> & neg) {

	// now search rows, followed by column.
	auto stime = getSysTime();
	pos.resize(mask.size());
	neg.resize(mask.size());

	FMT_ROOT_PRINT("mask size {}", mask.size());

	// walk through the mask
	size_t i = 0, j = 0;
	for (auto m : mask) {
		if (std::get<2>(m) == 0) {
			// FMT_ROOT_PRINT("getting pos value at [{} {}] to neg {} \n", std::get<0>(m), std::get<1>(m), j);
			neg.at(j) = values.at(std::get<0>(m), std::get<1>(m));
			++j;
		} else if (std::get<2>(m) == 1) {
			// FMT_ROOT_PRINT("getting pos value at [{} {}] to pos {} \n", std::get<0>(m), std::get<1>(m), i);
			pos.at(i) = values.at(std::get<0>(m), std::get<1>(m));
			++i;
		}
	}
	pos.resize(i);
	neg.resize(j);
	FMT_ROOT_PRINT("pos size {}, neg size {}", pos.size(), neg.size());
	
	auto etime = getSysTime();
	FMT_ROOT_PRINT("extracted values.  pos {} neg {} in {} sec\n", pos.size(), neg.size(), get_duration_s(stime, etime));

}




template <typename Kernel, typename Kernel2, typename T = double, typename L = char, typename O = double>
O compute_aupr_auroc(std::vector<std::tuple<size_t, size_t, int>> const & mask,
	splash::ds::aligned_matrix<T> const & vals, 
	Kernel const & auprkern, Kernel2 const & aurockern,
	splash::ds::aligned_vector<double> & pos, splash::ds::aligned_vector<double> & neg) {

	auto stime = getSysTime();
	select_values(mask, vals, pos, neg);
	auto etime = getSysTime();
	FMT_ROOT_PRINT("Select,truth,,{},sec\n", get_duration_s(stime, etime));

	stime = getSysTime();
	O aupr = auprkern(pos, neg);
	etime = getSysTime();
	FMT_ROOT_PRINT("Computed,aupr {},,{},sec\n", aupr, get_duration_s(stime, etime));

	stime = getSysTime();
	O auroc = aurockern(pos, neg);
	etime = getSysTime();
	FMT_ROOT_PRINT("Computed,auroc {},,{},sec\n", auroc, get_duration_s(stime, etime));

	return aupr;
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


	// =============== SETUP INPUT ===================
	// NOTE: input data is replicated on all MPI procs.
	using MatrixType = splash::ds::aligned_matrix<double>;
	// using VectorType = splash::ds::aligned_vector<std::pair<double, double>>;
	MatrixType input;
	std::vector<std::string> genes;
	std::vector<std::string> samples;
	
	auto stime = getSysTime();
	auto etime = getSysTime();

	stime = getSysTime();
	input = read_matrix<double>(common_params.input, "array", 
			common_params.num_vectors, common_params.vector_size,
			genes, samples, common_params.skip );
	etime = getSysTime();
	FMT_ROOT_PRINT("Loaded,INPUT,,{},sec\n", get_duration_s(stime, etime));


	stime = getSysTime();
	bool evaluate1 = app_params.groundtruth_list.length() > 0;
	bool evaluate2 = app_params.groundtruth_mat.length() > 0;
	std::vector<std::tuple<std::string, std::string, int>> truth;
	splash::ds::aligned_matrix<char> truth_mat;
	splash::ds::aligned_vector<double> pos;
	splash::ds::aligned_vector<double> neg;
	std::vector<std::pair<std::string, size_t>> dsrc_names;
	std::vector<std::pair<std::string, size_t>> dst_names;
	std::vector<std::tuple<size_t, size_t, int>> mask, mask2;
	mcp::kernel::aupr_kernel<double, char, double> auprkern;
	mcp::kernel::auroc_kernel<double, char, double> aurockern;

	size_t rows = common_params.num_vectors, columns = common_params.vector_size;
	std::vector<std::string> genes2;
	std::vector<std::string> samples2;

	if (evaluate1) {
		truth = read_groundtruth(app_params.groundtruth_list);
	} else if (evaluate2) {
		truth_mat = read_matrix<char>(app_params.groundtruth_mat, "array", 
			rows, columns,
			genes2, samples2);
	}

	etime = getSysTime();
	FMT_ROOT_PRINT("Loaded,truth,,{},sec\n", get_duration_s(stime, etime));


	if (mpi_params.rank == 0) {
		mpi_params.print("[PARAM] ");
		common_params.print("[PARAM] ");
		app_params.print("[PARAM] ");
	}


	if (evaluate1) {
		// onetime
		stime = getSysTime();
		sort_rowcol_names(genes, genes, dsrc_names, dst_names);
		etime = getSysTime();
		FMT_ROOT_PRINT("Computed,sorted names,,{},sec\n", get_duration_s(stime, etime));

		stime = getSysTime();
		mask = select_mask(truth, dsrc_names, dst_names);
		etime = getSysTime();
		FMT_ROOT_PRINT("Computed,truth mask,,{},sec\n", get_duration_s(stime, etime));

		// stime = getSysTime();
		// mask2 = select_mask2(truth, dsrc_names, dst_names);
		// etime = getSysTime();
		// FMT_ROOT_PRINT("Computed,truth mask2,,{},sec\n", get_duration_s(stime, etime));
		// end onetime.
	} else if (evaluate2) {
		stime = getSysTime();
		mask = select_lower_triangle(input.rows(), truth_mat);
		etime = getSysTime();
		FMT_ROOT_PRINT("Computed,truth mask mat,,{},sec\n", get_duration_s(stime, etime));
	}

	if (evaluate1 || evaluate2) {
		stime = getSysTime();
		compute_aupr_auroc(mask, input, auprkern, aurockern, pos, neg);
		etime = getSysTime();
		FMT_ROOT_PRINT("Computed,auprroc,,{},sec\n", get_duration_s(stime, etime));
	}


	return 0;
}
