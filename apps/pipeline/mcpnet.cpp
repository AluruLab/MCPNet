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


// combined, end to end, for generation and testing.
// this imple avoids file IO.

class app_parameters : public parameters_base {
	public:
		enum method_type : int { 
			UNUSED = 0, 
			MCP2 = 1, 
			MCP3 = 2, 
			MCP4 = 3, 
			MU_MCP = 4};
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
		bool clamped;
		std::vector<double> diagonals;

		app_parameters() : mi_computes(mi_method_type::AP), clamped(false) {}
		virtual ~app_parameters() {}

		virtual void config(CLI::App& app) {
			auto mi_group = app.add_option_group("MI");
			mi_group->add_option(app.add_option("--mi-method", mi_computes, "MI Algorithm: 0 (default) = Bspline, 1 = Adaptive Partitioning"));
			mi_group->add_option(app.add_option("--mi-file", mi_file, "precomputed MI file"));
			mi_group->require_option(1);

			auto comp_opt = app.add_option("-m,--method", computes, "Algorithm: MCP2=1, MCP3=2, MCP4=3, EnsembleMCP=4")->group("MCP");
			auto clamped_opt = app.add_flag("--clamped", clamped, "output is clampped")->group("MCP");
			clamped_opt->needs(comp_opt);

			app.add_option("--diagonal", diagonals, "Input MI matrix diagonal should be set as this value. default 0. if negative, use original MI")->group("MCP");

			app.add_option("-f,--coefficients", coeffs, "file with combo coefficients.  For ensemble MCP only.")->group("MCP");

			auto gt_list_opt = app.add_option("-g,--groundtruth-list", groundtruth_list, "filename of groundtruth edge list.  require just forward or reverse to be present. (+) -> 1, (-) -> 0, unknown -> -1");
			auto gt_mat_opt = app.add_option("-x,--groundtruth-matrix", groundtruth_mat, "filename of groundtruth matrix. require symmetric matrix. (+) -> 1, (-) -> 0, unknown -> -1");

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
					"unsupported"))))); 
			}
			FMT_ROOT_PRINT("{} MI method: {}\n", prefix, 
				(mi_computes == BSpline ? "bspline" : 
				(mi_computes == AP ? "adaptive partitioning with ranking" : "unknown"))); 
            FMT_ROOT_PRINT("{} MI file: {}\n", prefix, mi_file.c_str()); 
            FMT_ROOT_PRINT("{} coefficient input: {}\n", prefix, coeffs.c_str()); 
			FMT_ROOT_PRINT("{} MCP compute clamping output: {}\n", prefix, (clamped ? "true" : "false")); 
            FMT_ROOT_PRINT("{} groundtruth-list file: {}\n", prefix, groundtruth_list.c_str()); 
            FMT_ROOT_PRINT("{} groundtruth-matrix file: {}\n", prefix, groundtruth_mat.c_str()); 
			for (auto d : diagonals) {
				FMT_ROOT_PRINT("{} MI diagonal set to: {}\n", prefix, d); 
			}
		}
        
};

template <typename T, template <typename> class MatrixType>
void compute_mi(MatrixType<T> const & input, int const & bins, MatrixType<T> & dmi) {

	// 1. Scale : Vector -> Scaled Vector, partitioning
	splash::pattern::GlobalTransform<MatrixType<T>, 
		mcp::kernel::MinMaxScale<T>,
		MatrixType<T>> scaler;
	MatrixType<T> dscaled(input.rows(), input.columns());
	scaler(input, mcp::kernel::MinMaxScale<T>(), dscaled);

	// 
	// 2. Compute Weights : Vector -> Weight Vector
	// 
	int num_samples = (int) input.columns();
	splash::pattern::Transform<MatrixType<T>, 
		mcp::kernel::BSplineWeightsKernel<T>,
		MatrixType<T>> bspline_weights;
	MatrixType<T> dweighted(dscaled.rows(), 
		dscaled.columns() * bins + 1);
	bspline_weights(dscaled, 
		mcp::kernel::BSplineWeightsKernel<T>(
			bins,  // 10 bins
			3,   // spline order
			num_samples),
		dweighted);	

	// --------------- bspline ----------------------
	auto stime = getSysTime();
	MatrixType<T> weighted = dweighted.allgather();
	auto etime = getSysTime();
	FMT_ROOT_PRINT("Comm,weighted allgather,,{},sec\n", get_duration_s(stime, etime));

	// 4. Inner Product :  Weight Vector, Weight Vector -> MI
	// ---- create a VV2S kernel
	splash::pattern::InnerProduct<
		MatrixType<T>, 
		mcp::kernel::BSplineMIKernel<T>,
		MatrixType<T>, true > mi_proc;		
	mi_proc(weighted, weighted, 
		mcp::kernel::BSplineMIKernel<T>(
			bins,
			num_samples),
		dmi);
}


template <typename T, template <typename> class MatrixType>
void compute_ap_mi(MatrixType<T> const & input,  MatrixType<T> & dmi, bool const & _add_noise = false) {

	// 1. Rank transform
	// ------------ perturb -----------
	MatrixType<T> noisy(input.rows(), input.columns());
	if (_add_noise) {
		auto stime = getSysTime();
		// ---- create a VV2S kernel
		using NoiseKernelType = ::mcp::kernel::add_noise<T, std::uniform_real_distribution<T>, std::default_random_engine>;
		splash::kernel::random_number_generator<> gen;
		NoiseKernelType noise_adder(gen, 0.0, 0.00001);
		noisy.resize(input.rows(), input.columns()); 
		noise_adder(input, noisy);

		auto etime = getSysTime();
		FMT_ROOT_PRINT("Rank Transformed in {} sec\n", get_duration_s(stime, etime));

	} else {
		noisy = input;
	}
	
	// ------------ rank -----------
	auto stime = getSysTime();
	// ---- create a VV2S kernel
	using RankKernelType = splash::kernel::Rank<T, size_t, 0, false>;  // descending
	splash::pattern::Transform<MatrixType<T>, 
		RankKernelType,
		splash::ds::aligned_matrix<size_t>> normalizer;
	RankKernelType ranker;
	splash::ds::aligned_matrix<size_t> ranked(noisy.rows(), noisy.columns()); 
	normalizer(noisy, ranker, ranked);

	auto etime = getSysTime();
	FMT_ROOT_PRINT("Rank Transformed in {} sec\n", get_duration_s(stime, etime));

	// 2. histogram
	// ------------ scale -----------
	stime = getSysTime();
	// ---- create a VV2S kernel
	using HistogramKernelType = mcp::kernel::IntegralCumulativeHistogramKernel<size_t, 0>;
	splash::pattern::Transform<splash::ds::aligned_matrix<size_t>, 
		HistogramKernelType,
		splash::ds::aligned_matrix<size_t>> histographer;
	HistogramKernelType counter;
	splash::ds::aligned_matrix<size_t> histos(ranked.rows(), ranked.columns() + 1); 
	histographer(ranked, counter, histos);

	etime = getSysTime();
	FMT_ROOT_PRINT("Rank Histogram in {} sec\n", get_duration_s(stime, etime));

	// 4. Inner Product :  Weight Vector, Weight Vector -> MI
	stime = getSysTime();
	// ---- create a VV2S kernel
	using MIKernelType = mcp::kernel::AdaptivePartitionRankMIKernel2<size_t, T>;
	splash::pattern::InnerProduct<
		splash::ds::aligned_matrix<size_t>, 
		MIKernelType,
		MatrixType<T>, true > adaptive_mi;
	MIKernelType mi_op(histos.data(), histos.column_bytes());
	adaptive_mi(ranked, ranked, mi_op, dmi);

	etime = getSysTime();
	FMT_ROOT_PRINT("Compute Adpative MI in {} sec\n", get_duration_s(stime, etime));

}


template <typename T, template <typename> class MatrixType>
void compute_mcp2(MatrixType<T> const & genes_to_genes, MatrixType<T> & dmcp2) {

	// first max{min}
	MatrixType<T> dmaxmin1; 
	using MXKernelType = mcp::kernel::mcp2_maxmin_kernel<T, false, true>;
	MXKernelType mxkernel;  // clamped.
	::splash::pattern::InnerProduct<MatrixType<T>, MXKernelType, MatrixType<T>, true> mxgen_s;
	dmaxmin1.resize(genes_to_genes.rows(), genes_to_genes.columns());
	mxgen_s(genes_to_genes, genes_to_genes, mxkernel, dmaxmin1);

	splash::utils::partition<size_t> g2g_part = splash::utils::partition<size_t>::make_partition(dmaxmin1.rows());
	splash::utils::partition<size_t> dmcp_part(0, dmaxmin1.rows(), 0);

	dmcp2.resize(dmaxmin1.rows(), dmaxmin1.columns());
	// correlation close to 0 is bad.
	using KernelType = mcp::kernel::ratio_kernel<T, T>;
	KernelType kernel;  // clamped.
	::splash::pattern::BinaryOp<MatrixType<T>, MatrixType<T>, KernelType, MatrixType<T>> tolgen;
	tolgen(genes_to_genes, g2g_part, dmaxmin1, dmcp_part, kernel, dmcp2, dmcp_part);
}

template <typename T, template <typename> class MatrixType>
void compute_mcp3(MatrixType<T> const & genes_to_genes, MatrixType<T> & dmcp3) {

	using MXKernelType = mcp::kernel::mcp2_maxmin_kernel<T, false, true>;
	MXKernelType mxkernel;  // clamped.
	::splash::pattern::InnerProduct<MatrixType<T>, MXKernelType, MatrixType<T>, false> mxgen;
	::splash::pattern::InnerProduct<MatrixType<T>, MXKernelType, MatrixType<T>, true> mxgen_s;

	// first max{min}
	// compute leg2 to 3.  OUTPUT MUST BE input.rows() x input.columns().  Potentially skipping rows. 
	MatrixType<T> dmaxmin1(genes_to_genes.rows(), genes_to_genes.columns());
	mxgen_s(genes_to_genes, genes_to_genes, mxkernel, dmaxmin1);

	auto stime = getSysTime();
	MatrixType<T> maxmin1 = dmaxmin1.allgather();
	auto etime = getSysTime();
	FMT_ROOT_PRINT("Comm,maxmin1 allgather,,{},sec\n", get_duration_s(stime, etime));

	// second max{min}
	MatrixType<T> dmaxmin2(genes_to_genes.rows(), maxmin1.columns());
	mxgen(genes_to_genes, maxmin1, mxkernel, dmaxmin2);  // NOTE this is not symmetric? previous logic had that assumption.
	
	splash::utils::partition<size_t> g2g_part = splash::utils::partition<size_t>::make_partition(dmaxmin2.rows());
	splash::utils::partition<size_t> dmcp_part(0, dmaxmin2.rows(), 0);

	dmcp3.resize(dmaxmin2.rows(), dmaxmin2.columns());

	// correlation close to 0 is bad.
	using KernelType = mcp::kernel::ratio_kernel<T, T>;
	KernelType kernel;  // clamped.
	::splash::pattern::BinaryOp<MatrixType<T>, MatrixType<T>, KernelType, MatrixType<T>> tolgen;
	tolgen(genes_to_genes, g2g_part, dmaxmin2, dmcp_part, kernel, dmcp3, dmcp_part);
}

template <typename T, template <typename> class MatrixType>
void compute_mcp4(MatrixType<T> const & genes_to_genes, MatrixType<T> & dmcp4) {

	using MXKernelType = mcp::kernel::mcp2_maxmin_kernel<T, false, true>;
	MXKernelType mxkernel;  // clamped.
	::splash::pattern::InnerProduct<MatrixType<T>, MXKernelType, MatrixType<T>, true> mxgen_s;

	// first max{min}
	MatrixType<T> dmaxmin1(genes_to_genes.rows(), genes_to_genes.columns());
	mxgen_s(genes_to_genes, genes_to_genes, mxkernel, dmaxmin1);
	
	auto stime = getSysTime();
	MatrixType<T> maxmin1 = dmaxmin1.allgather();
	auto etime = getSysTime();
	FMT_ROOT_PRINT("Comm,maxmin1 allgather,,{},sec\n", get_duration_s(stime, etime));

	// second max{min},  same as first.
	MatrixType<T> dmaxmin2(genes_to_genes.rows(), genes_to_genes.columns());
	mxgen_s(maxmin1, maxmin1, mxkernel, dmaxmin2);

	splash::utils::partition<size_t> g2g_part = splash::utils::partition<size_t>::make_partition(dmaxmin2.rows());
	splash::utils::partition<size_t> dmcp_part(0, dmaxmin2.rows(), 0);

	dmcp4.resize(dmaxmin2.rows(), dmaxmin2.columns());

	// correlation close to 0 is bad.
	using KernelType = mcp::kernel::ratio_kernel<T, T>;
	KernelType kernel;  // clamped.
	::splash::pattern::BinaryOp<MatrixType<T>, MatrixType<T>, KernelType, MatrixType<T>> tolgen;
	tolgen(genes_to_genes, g2g_part, dmaxmin2, dmcp_part, kernel, dmcp4, dmcp_part);		
}



template <typename T, template <typename> class MatrixType>
void compute_maxmins(MatrixType<T> const & genes_to_genes, 
	MatrixType<T> & dmaxmin1, 
	MatrixType<T> & dmaxmin2, 
	MatrixType<T> & dmaxmin3 ) {

	using MXKernelType = mcp::kernel::mcp2_maxmin_kernel<T, false, true>;
	MXKernelType mxkernel;  // clamped.
	::splash::pattern::InnerProduct<MatrixType<T>, MXKernelType, MatrixType<T>, false> mxgen;
	::splash::pattern::InnerProduct<MatrixType<T>, MXKernelType, MatrixType<T>, true> mxgen_s;

	// first max{min}
	mxgen_s(genes_to_genes, genes_to_genes, mxkernel, dmaxmin1);

	auto stime = getSysTime();	
	MatrixType<T> maxmin1 = dmaxmin1.allgather();
	auto etime = getSysTime();
	FMT_ROOT_PRINT("Comm,maxmin1 allgather,,{},sec\n", get_duration_s(stime, etime));

	// second max{min},  same as first.
	mxgen(genes_to_genes, maxmin1, mxkernel, dmaxmin2);

	// third max{min},  same as first.
	mxgen_s(maxmin1, maxmin1, mxkernel, dmaxmin3);
}


//  combined entire linear combination as a single kernel.  much faster.
template <typename T, template <typename> class MatrixType>
void compute_combo(MatrixType<T> const & genes_to_genes, 
	MatrixType<T> const & dmaxmin1, 
	MatrixType<T> const & dmaxmin2, 
	MatrixType<T> const & dmaxmin3,
	double const * cs,
	MatrixType<T> & dcombo ) {

	// compute linear combination.
	splash::utils::partitioner1D<PARTITION_EQUAL> partitioner;
	splash::utils::partition<size_t> mpi_g2g_parts = splash::utils::partition<size_t>::make_partition(dmaxmin1.rows());
	splash::utils::partition<size_t> mpi_mxmn_parts(0, dmaxmin1.rows(), 0);

	dcombo.resize(dmaxmin1.rows(), dmaxmin1.columns());

#ifdef USE_OPENMP
#pragma omp parallel
	{
		int threads = omp_get_num_threads();
		int thread_id = omp_get_thread_num();
#else 
	{
		int threads = 1;
		int thread_id = 0;
#endif

		// partition the local 2D tiles.  omp_tile_parts.offset is local to this processor.
		splash::utils::partition<size_t> omp_g2g_part = partitioner.get_partition(mpi_g2g_parts, threads, thread_id);
		splash::utils::partition<size_t> omp_mxmn_part = partitioner.get_partition(mpi_mxmn_parts, threads, thread_id);

		// FMT_PRINT_RT("coeff ptr : {}\n", reinterpret_cast<void const *>(cs));

		// iterate over rows.
		size_t gid = omp_g2g_part.offset;
		size_t mid = omp_mxmn_part.offset;
		
		double a = cs[0];
		double b=cs[1];
		double c=cs[2];
		double d=cs[3];
		double maxmins = 0;

		for (size_t i = 0; i < omp_mxmn_part.size; ++i, ++gid, ++mid) {
			T const * dmi = genes_to_genes.data(gid);
			T const * dmm1 = dmaxmin1.data(mid);
			T const * dmm2 = dmaxmin2.data(mid);
			T const * dmm3 = dmaxmin3.data(mid);
			T * out = dcombo.data(mid);
			for (size_t j = 0; j < dmaxmin1.columns(); ++j) {
				maxmins = 0.0;
				if (a == 1.0) maxmins += dmi[j];  else if (a != 0.0) maxmins += a * dmi[j];
				if (b == 1.0) maxmins += dmm1[j]; else if (b != 0.0) maxmins += b * dmm1[j];
				if (c == 1.0) maxmins += dmm2[j]; else if (c != 0.0) maxmins += c * dmm2[j];
				if (d == 1.0) maxmins += dmm3[j]; else if (d != 0.0) maxmins += d * dmm3[j];

				// note: if a == 1.0 and b, c, d == 0, then all output values will be 1.0
				out[j] = (std::abs(dmi[j]) < std::numeric_limits<T>::epsilon()) ? 0.0 :
					((std::abs(maxmins) < std::numeric_limits<T>::epsilon()) ? std::numeric_limits<T>::max() :
					dmi[j] / maxmins);
			}
		}
	}

}

// this is not scaling.  not sure why.
template <typename T, template <typename> class MatrixType>
void compute_stouffer(MatrixType<T> const & dinput, MatrixType<T> & dstouffer) {
	using KernelType = mcp::kernel::stouffer_vector_kernel<T>;
	KernelType transform;
	using ReducType = splash::kernel::GaussianParamsExclude1<T, T, true>;
	ReducType reduc;
	splash::pattern::ReduceTransform<MatrixType<T>, 
		ReducType, KernelType,
		MatrixType<T>> transformer;
	transformer(dinput, reduc, transform, dstouffer);
}


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

	stime = getSysTime();
	bool eval_list = app_params.groundtruth_list.length() > 0;
	bool eval_mat = app_params.groundtruth_mat.length() > 0;
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

	if (eval_list) {
		truth = read_groundtruth(app_params.groundtruth_list);
	} else if (eval_mat) {
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

	std::string output_prefix = common_params.output;
	if (app_params.mi_computes == app_parameters::mi_method_type::BSpline)
		output_prefix.append("_bs");
	else 
		output_prefix.append("_ap");

	std::ostringstream oss;
	oss.str("");
	oss.clear();


	// =============== MI ===================
	MatrixType dmi;
	if (app_params.mi_file.length() == 0) {
		dmi.resize(input.rows(), input.columns());
		stime = getSysTime();
		if (app_params.mi_computes == app_parameters::mi_method_type::BSpline)
			compute_mi(input, app_params.num_bins, dmi);
		else  
			compute_ap_mi(input, dmi, true);  // add noise.
		etime = getSysTime();
		FMT_ROOT_PRINT("Computed,MI,,{},sec\n", get_duration_s(stime, etime));
	} else {
		// distribute the mi to dmi.
		dmi = mi.scatter(0);
	}

	if (eval_list) {
		// onetime
		stime = getSysTime();
		sort_rowcol_names(dmi.rows(), genes, genes, dsrc_names, dst_names);
		etime = getSysTime();
		FMT_ROOT_PRINT("Computed,sorted names,,{},sec\n", get_duration_s(stime, etime));

		stime = getSysTime();
		mask = select_mask(truth, dsrc_names, dst_names);
		etime = getSysTime();
		FMT_ROOT_PRINT("Computed,truth mask,,{},sec\n", get_duration_s(stime, etime));

	} else if (eval_mat) {
		stime = getSysTime();
		mask = select_lower_triangle(dmi.rows(), truth_mat);
		etime = getSysTime();
		FMT_ROOT_PRINT("Computed,truth mask mat,,{},sec\n", get_duration_s(stime, etime));
	}

	if (eval_list || eval_mat) {
		stime = getSysTime();
		compute_aupr_auroc(mask, dmi, auprkern, aurockern, pos, neg);
		etime = getSysTime();
		FMT_ROOT_PRINT("Computed,auprroc,,{},sec\n", get_duration_s(stime, etime));
	}

	stime = getSysTime();
	oss << output_prefix << ".mi.h5";
	write_matrix_distributed(oss.str(), "array", genes, genes, dmi);
	oss.str("");
	oss.clear();
	etime = getSysTime();
	FMT_ROOT_PRINT("Wrote,MI,,{},sec\n", get_duration_s(stime, etime));
	FMT_FLUSH();

	// =============== MCP 2, 3, 4 and ensemble ========
	std::unordered_set<app_parameters::method_type> comps;
	for (auto c : app_params.computes) {
		comps.insert(c);
	}

	// gather
	if (app_params.mi_file.length() == 0) {
		stime = getSysTime();
		mi = dmi.allgather();
		etime = getSysTime();
		FMT_ROOT_PRINT("Gathered MI,,{},sec\n", get_duration_s(stime, etime));
	}	
	stime = getSysTime();
	// if diagonal is specified, then use it.
	bool reset_diag = app_params.diagonals.size() > 0;
	if (reset_diag) {
		double diagonal = app_params.diagonals[0];
		size_t mc = std::min(mi.rows(), mi.columns());
		for (size_t i = 0; i < mc; ++i) {
			mi.at(i, i) = diagonal;
		}
	}
	etime = getSysTime();
	FMT_ROOT_PRINT("Set MI diagonal {},sec\n", get_duration_s(stime, etime));

	MatrixType dratio;
	if (comps.find(app_parameters::method_type::MCP2) != comps.end()) {
		// ------ ratio 1 --------
		stime = getSysTime();
		compute_mcp2(mi, dratio);
		etime = getSysTime();
		FMT_ROOT_PRINT("Computed,MCP2,,{},sec\n", get_duration_s(stime, etime));


		if (eval_list || eval_mat) {
			stime = getSysTime();
			compute_aupr_auroc(mask, dratio, auprkern, aurockern, pos, neg);
			etime = getSysTime();
			FMT_ROOT_PRINT("Computed,auprroc,,{},sec\n", get_duration_s(stime, etime));

		}


		stime = getSysTime();
		oss << output_prefix << ".mcp2.h5";
		write_matrix_distributed(oss.str(), "array", genes, genes, dratio);
		oss.str("");
		oss.clear();
		etime = getSysTime();
		FMT_ROOT_PRINT("Wrote,MCP2,,{},sec\n", get_duration_s(stime, etime));
		FMT_FLUSH();
	} 
	
	dratio.zero();
	if (comps.find(app_parameters::method_type::MCP3) != comps.end()) {
		// ------ ratio 1 --------
		stime = getSysTime();
		compute_mcp3(mi, dratio);
		etime = getSysTime();
		FMT_ROOT_PRINT("Computed,MCP3,,{},sec\n", get_duration_s(stime, etime));
		
		if (eval_list || eval_mat) {
			stime = getSysTime();
			compute_aupr_auroc(mask, dratio, auprkern, aurockern, pos, neg);
			etime = getSysTime();
			FMT_ROOT_PRINT("Computed,auprroc,,{},sec\n", get_duration_s(stime, etime));
		}

		stime = getSysTime();
		oss << output_prefix << ".mcp3.h5";
		write_matrix_distributed(oss.str(), "array", genes, genes, dratio);
		oss.str("");
		oss.clear();
		etime = getSysTime();
		FMT_ROOT_PRINT("Wrote,MCP3,,{},sec\n", get_duration_s(stime, etime));
		FMT_FLUSH();
	} 

	dratio.zero();
	if (comps.find(app_parameters::method_type::MCP4) != comps.end()) {
		// ------ ratio 1 --------
		stime = getSysTime();
		compute_mcp4(mi, dratio);
		etime = getSysTime();
		FMT_ROOT_PRINT("Computed,MCP4,,{},sec\n", get_duration_s(stime, etime));

		if (eval_list || eval_mat) {
			stime = getSysTime();
			compute_aupr_auroc(mask, dratio, auprkern, aurockern, pos, neg);
			etime = getSysTime();
			FMT_ROOT_PRINT("Computed,auprroc,,{},sec\n", get_duration_s(stime, etime));
		}

		stime = getSysTime();
		oss << output_prefix << ".mcp4.h5";
		write_matrix_distributed(oss.str(), "array", genes, genes, dratio);
		oss.str("");
		oss.clear();
		etime = getSysTime();
		FMT_ROOT_PRINT("Wrote,MCP4,,{},sec\n", get_duration_s(stime, etime));
		FMT_FLUSH();
	} 
	
	dratio.zero();
	// ============= compute combos ====================
	if (comps.find(app_parameters::method_type::MU_MCP) != comps.end()) {
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
		compute_maxmins(mi, dmaxmin1, dmaxmin2, dmaxmin3);
		etime = getSysTime();
		FMT_ROOT_PRINT("Computed,MAXMIN 1 2 3,,{},sec\n", get_duration_s(stime, etime));


		// ----------- compute the combos
		auto stime_all = getSysTime();
		MatrixType dratiost;
		double* coeff = nullptr;
		oss << std::fixed << std::setprecision(3);

		// parameters for eval
		double max_aupr = 0, aupr;
		double max_saupr = 0, saupr;
		size_t co = 0, sco = 0;

		for (size_t i = 0; i < rs; ++i) {
			coeff = coeffs.data(i);

			// ------ combo ratio --------
			stime = getSysTime();
			compute_combo(mi, dmaxmin1, dmaxmin2, dmaxmin3, coeff, dratio);
			etime = getSysTime();
			FMT_ROOT_PRINT("Computed,Combo,({}_{}_{}_{}),{},sec\n", 
				coeff[0], coeff[1], coeff[2], coeff[3], get_duration_s(stime, etime));

			if (eval_list || eval_mat) {
				stime = getSysTime();
				aupr = compute_aupr_auroc(mask, dratio, auprkern, aurockern, pos, neg);

				// save coeff if aupr is max.
				if (aupr > max_aupr) {
					max_aupr = aupr;
					co = i;
				}
				etime = getSysTime();
				FMT_ROOT_PRINT("Computed,Combo AUPR,({}_{}_{}_{}),{},sec\n", 
					coeff[0], coeff[1], coeff[2], coeff[3], get_duration_s(stime, etime));
			} else {
				stime = getSysTime();
				oss << output_prefix << ".mcp_" << coeff[0] <<
					"_" << coeff[1] << "_" << coeff[2] << "_" << coeff[3] << ".h5";
				write_matrix_distributed(oss.str(), "array", genes, genes, dratio);
				oss.str("");
				oss.clear();
				etime = getSysTime();
				FMT_ROOT_PRINT("Wrote,Combo,({}_{}_{}_{}),{},sec\n", 
					coeff[0], coeff[1], coeff[2], coeff[3], get_duration_s(stime, etime));
			}

			// ------- stouffer --------
			stime = getSysTime();
			compute_stouffer(dratio, dratiost); 
			etime = getSysTime();
			FMT_ROOT_PRINT("Computed,Combo Stouffer,({}_{}_{}_{}),{},sec\n", 
				coeff[0], coeff[1], coeff[2], coeff[3], get_duration_s(stime, etime));

			// write out again.
			if (eval_list || eval_mat) {
				
			//	for (size_t i = 0; i < dratiost.columns(); ++i) {
			//		FMT_ROOT_PRINT("{}, ", dratiost.at(0, i) );
			//	}

				stime = getSysTime();
				saupr = compute_aupr_auroc(mask, dratiost, auprkern, aurockern, pos, neg);


				// save coeff if aupr is max.
				if (saupr > max_saupr) {
					max_saupr = saupr;
					sco = i;
				}
				etime = getSysTime();
				FMT_ROOT_PRINT("Computed,Combo Stouffer AUPR,({}_{}_{}_{}),{},sec\n", 
					coeff[0], coeff[1], coeff[2], coeff[3], get_duration_s(stime, etime));
			} else {
				stime = getSysTime();
				oss << output_prefix << ".mcpst_" << coeff[0] <<
					"_" << coeff[1] << "_" << coeff[2] << "_" << coeff[3] << ".h5";
				write_matrix_distributed(oss.str(), "array", genes, genes, dratiost);
				oss.str("");
				oss.clear();

				etime = getSysTime();
				FMT_ROOT_PRINT("Wrote,Combo Stouffer,({}_{}_{}_{}),{},sec\n", 
					coeff[0], coeff[1], coeff[2], coeff[3], get_duration_s(stime, etime));
			}
		}

		if (eval_list || eval_mat) {
			// ------ combo ratio --------
			stime = getSysTime();
			compute_combo(mi, dmaxmin1, dmaxmin2, dmaxmin3, coeffs.data(co), dratio);
			etime = getSysTime();
			FMT_ROOT_PRINT("Computed,MAX Combo aupr {},({}_{}_{}_{}),{},sec\n", 
				max_aupr, coeffs.at(co, 0), coeffs.at(co, 1), coeffs.at(co, 2), coeffs.at(co, 3), get_duration_s(stime, etime));
			stime = getSysTime();
			compute_aupr_auroc(mask, dratio, auprkern, aurockern, pos, neg);
			etime = getSysTime();
			FMT_ROOT_PRINT("Computed,MAX Combo aupr auroc,,{},sec\n", get_duration_s(stime, etime));


			stime = getSysTime();
			// write out results with the max AUPR for regular and stouffer.
			oss << output_prefix << ".mcp_" << coeffs.at(co, 0) <<
				"_" << coeffs.at(co, 1) << "_" << coeffs.at(co, 2) << "_" << coeffs.at(co, 3) << ".h5";
			write_matrix_distributed(oss.str(), "array", genes, genes, dratio);
			oss.str("");
			oss.clear();
			etime = getSysTime();
			FMT_ROOT_PRINT("Wrote,MAX Combo,({}_{}_{}_{}),{},sec\n", 
				coeffs.at(co, 0), coeffs.at(co, 1), coeffs.at(co, 2), coeffs.at(co, 3), get_duration_s(stime, etime));


			// ------- stouffer --------
			stime = getSysTime();
			compute_combo(mi, dmaxmin1, dmaxmin2, dmaxmin3, coeffs.data(sco), dratio);
			compute_stouffer(dratio, dratiost); 
			etime = getSysTime();
			FMT_ROOT_PRINT("Computed,MAX Combo AND Stouffer aupr {},({}_{}_{}_{}),{},sec\n", 
				max_saupr, coeffs.at(sco, 0), coeffs.at(sco, 1), coeffs.at(sco, 2), coeffs.at(sco, 3), get_duration_s(stime, etime));
			stime = getSysTime();
			compute_aupr_auroc(mask, dratiost, auprkern, aurockern, pos, neg);
			etime = getSysTime();
			FMT_ROOT_PRINT("Computed,MAX Combo AND Stouffer aupr auroc,,{},sec\n", get_duration_s(stime, etime));

			stime = getSysTime();
			// write out results with the max AUPR for regular and stouffer.
			oss << output_prefix << ".mcpst_" << coeffs.at(sco, 0) <<
				"_" << coeffs.at(sco, 1) << "_" << coeffs.at(sco, 2) << "_" << coeffs.at(sco, 3) << ".h5";
			write_matrix_distributed(oss.str(), "array", genes, genes, dratiost);
			oss.str("");
			oss.clear();
			etime = getSysTime();
			FMT_ROOT_PRINT("Wrote,MAX Combo Stouffer,({}_{}_{}_{}),{},sec\n", 
				coeffs.at(sco, 0), coeffs.at(sco, 1), coeffs.at(sco, 2), coeffs.at(sco, 3), get_duration_s(stime, etime));

		}

		auto etime_all = getSysTime();
		FMT_ROOT_PRINT("ALL Combos in {} sec\n", get_duration_s(stime_all, etime_all));
		FMT_FLUSH();
	} 

	return 0;
}
