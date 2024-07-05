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

#pragma once

#include <sstream>
#include <fstream>
#include <iostream>
#include <iomanip> // ostringstream
#include <vector>
#include <unordered_set>
#include <string>  //getline
#include <random>

#include "splash/utils/benchmark.hpp"
#include "splash/utils/report.hpp"
#include "splash/ds/aligned_matrix.hpp"
#include "splash/patterns/pattern.hpp"

#include "mcp/correlation/BSplineMI.hpp"
#include "mcp/correlation/AdaptivePartitioningMI.hpp"
#include "splash/transform/rank.hpp"

#include "mcp/filter/mcp.hpp"
#include "mcp/filter/dpi.hpp"
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
		MatrixType<T>, true > bspline_mi;		
	bspline_mi(weighted, weighted, 
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
void compute_ratio(MatrixType<T> const & mi, MatrixType<T> const & dmaxmin, MatrixType<T> & dmcp) {

	splash::utils::partition<size_t> global_part = splash::utils::partition<size_t>::make_partition(dmaxmin.rows());
	splash::utils::partition<size_t> local_part(0, dmaxmin.rows(), 0);

	dmcp.resize(dmaxmin.rows(), dmaxmin.columns());
	// correlation close to 0 is bad.
	using KernelType = mcp::kernel::ratio_kernel<T, T>;
	KernelType kernel;  // clamped.
	::splash::pattern::BinaryOp<MatrixType<T>, MatrixType<T>, KernelType, MatrixType<T>> gen;
	gen(mi, global_part, dmaxmin, local_part, kernel, dmcp, local_part);
}



template <typename T, template <typename> class MatrixType>
void compute_maxmin1(MatrixType<T> const & genes_to_genes, MatrixType<T> & dmaxmin) {
	// first max{min}
	using MXKernelType = mcp::kernel::mcp2_maxmin_kernel<T, false, true>;
	MXKernelType mxkernel;  // clamped.
	::splash::pattern::InnerProduct<MatrixType<T>, MXKernelType, MatrixType<T>, true> mxgen_s;
	dmaxmin.resize(genes_to_genes.rows(), genes_to_genes.columns());
	mxgen_s(genes_to_genes, genes_to_genes, mxkernel, dmaxmin);
}

template <typename T, template <typename> class MatrixType>
void compute_mcp2(MatrixType<T> const & genes_to_genes, MatrixType<T> & dmcp) {

	MatrixType<T> dmaxmin;
	compute_maxmin1(genes_to_genes, dmaxmin);

	compute_ratio(genes_to_genes, dmaxmin, dmcp);
}


template <typename T, template <typename> class MatrixType>
void compute_maxmin1_tfs(MatrixType<T> const & tfs_to_genes, 
	MatrixType<T> const & genes_to_genes, int const & tf_gene_transition,
	std::vector<T> const & tfs, MatrixType<T> & dmaxmin) {
	
	if ((tf_gene_transition < 0) || (tf_gene_transition > 1)) return;

	// first max{min}
	dmaxmin.resize(tfs_to_genes.rows(), genes_to_genes.columns()); 
	if (tf_gene_transition == 1) {
		// tfs-tfs-genes
		using MXKernelType = mcp::kernel::mcp2_maxmin_kernel<T, true, true>;
		MXKernelType mxkernel(tfs);  // clamped.
		::splash::pattern::InnerProduct<MatrixType<T>, MXKernelType, MatrixType<T>, false> mxgen;
		// tf-gen transition on second gene.
		mxgen(tfs_to_genes, genes_to_genes, mxkernel, dmaxmin);
	} else  {
		// tfs-genes-genes
		using MXKernelType = mcp::kernel::mcp2_maxmin_kernel<T, false, true>;
		MXKernelType mxkernel;  // clamped.
		::splash::pattern::InnerProduct<MatrixType<T>, MXKernelType, MatrixType<T>, false> mxgen;
		// default, edge has tf to gene transition.
		mxgen(tfs_to_genes, genes_to_genes, mxkernel, dmaxmin);
	}
}

template <typename T, template <typename> class MatrixType>
void compute_mcp2_tfs(MatrixType<T> const & tfs_to_genes, 
	MatrixType<T> const & genes_to_genes, int const & tf_gene_transition,
	std::vector<T> const & tfs, MatrixType<T> & dmcp) {

	MatrixType<T> dmaxmin;
	compute_maxmin1_tfs(tfs_to_genes, genes_to_genes, tf_gene_transition, tfs, dmaxmin);
	
	compute_ratio(tfs_to_genes, dmaxmin, dmcp);
}



template <typename T, template <typename> class MatrixType>
void compute_maxmin2(MatrixType<T> const & genes_to_genes, MatrixType<T> & dmaxmin) {

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
	dmaxmin.resize(genes_to_genes.rows(), maxmin1.columns());
	mxgen(genes_to_genes, maxmin1, mxkernel, dmaxmin);  // NOTE this is not symmetric? previous logic had that assumption.
}

template <typename T, template <typename> class MatrixType>
void compute_mcp3(MatrixType<T> const & genes_to_genes, MatrixType<T> & dmcp) {

	MatrixType<T> dmaxmin;
	compute_maxmin2(genes_to_genes, dmaxmin);
	
	compute_ratio(genes_to_genes, dmaxmin, dmcp);
}


template <typename T, template <typename> class MatrixType>
void compute_maxmin2_tfs(MatrixType<T> const & tfs_to_genes, MatrixType<T> const & genes_to_genes, int const & tf_gene_transition,
	std::vector<T> const & tfs, MatrixType<T> & dmaxmin) {

	if ((tf_gene_transition < 0) || (tf_gene_transition > 2)) return;


	using MaskedMXKernelType = mcp::kernel::mcp2_maxmin_kernel<T, true, true>;
	MaskedMXKernelType maskedmxkernel(tfs);  // clamped.
	using MXKernelType = mcp::kernel::mcp2_maxmin_kernel<T, false, true>;
	MXKernelType mxkernel;  // clamped.

	// first max{min}
	// compute leg2 to 3.  OUTPUT MUST BE input.rows() x input.columns().  Potentially skipping rows. 
	MatrixType<T> dmaxmin1(genes_to_genes.rows(), genes_to_genes.columns());
	if (tf_gene_transition == 2) {  // leg 2.
		::splash::pattern::InnerProduct<MatrixType<T>, MaskedMXKernelType, MatrixType<T>, true> mxgen;
		// tf-(tf-tf-gen) transition.  make gen-tf-gen.
		mxgen(genes_to_genes, genes_to_genes, maskedmxkernel, dmaxmin1);
	} else {  // leg 0, 1, and none.
		::splash::pattern::InnerProduct<MatrixType<T>, MXKernelType, MatrixType<T>, true> mxgen;
		// tf-(tf-gen-gen), tf-(gene-gene-gene), gene-(gene-gene-gene)  make gene-gene-gene.  
		mxgen(genes_to_genes, genes_to_genes, mxkernel, dmaxmin1);
	}
	auto stime = getSysTime();
	MatrixType<T> maxmin1 = dmaxmin1.allgather();
	auto etime = getSysTime();
	FMT_ROOT_PRINT("Comm,maxmin1 allgather,,{},sec\n", get_duration_s(stime, etime));

	// second max{min}
	dmaxmin.resize(tfs_to_genes.rows(), maxmin1.columns());
	if ((tf_gene_transition == 2) || (tf_gene_transition == 1)) {
		// tf-tf-gene-gene, or tf-tf-tf-gene
		::splash::pattern::InnerProduct<MatrixType<T>, MaskedMXKernelType, MatrixType<T>, false> mxgen;
		mxgen(tfs_to_genes, maxmin1, maskedmxkernel, dmaxmin);
	} else {  // leg 0.
		// tf-(gene-gene-gene), edge has tf to gene transition.
		::splash::pattern::InnerProduct<MatrixType<T>, MXKernelType, MatrixType<T>, false> mxgen;
		mxgen(tfs_to_genes, maxmin1, mxkernel, dmaxmin);
	}

}


template <typename T, template <typename> class MatrixType>
void compute_mcp3_tfs(MatrixType<T> const & tfs_to_genes, MatrixType<T> const & genes_to_genes, int const & tf_gene_transition,
	std::vector<T> const & tfs, MatrixType<T> & dmcp) {

	MatrixType<T> dmaxmin;
	compute_maxmin2_tfs(tfs_to_genes, genes_to_genes, tf_gene_transition, tfs, dmaxmin);
	
	compute_ratio(tfs_to_genes, dmaxmin, dmcp);

}

template <typename T, template <typename> class MatrixType>
void compute_maxmin3(MatrixType<T> const & genes_to_genes, MatrixType<T> & dmaxmin) {

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
	dmaxmin.resize(genes_to_genes.rows(), genes_to_genes.columns());
	mxgen_s(maxmin1, maxmin1, mxkernel, dmaxmin);

}

template <typename T, template <typename> class MatrixType>
void compute_mcp4(MatrixType<T> const & genes_to_genes, MatrixType<T> & dmcp) {

	MatrixType<T> dmaxmin;
	compute_maxmin3(genes_to_genes, dmaxmin);
	
	compute_ratio(genes_to_genes, dmaxmin, dmcp);
}

template <typename T, template <typename> class MatrixType>
void compute_maxmin3_tfs(MatrixType<T> const & tfs_to_genes, MatrixType<T> const & genes_to_genes, int const & tf_gene_transition,
	std::vector<T> const & tfs, MatrixType<T> & dmaxmin) {

	if ((tf_gene_transition < 0) || (tf_gene_transition > 3)) return;

	using MaskedMXKernelType = mcp::kernel::mcp2_maxmin_kernel<T, true, true>;
	MaskedMXKernelType maskedmxkernel(tfs);  // when masking, some columns of first matrix and rows of second are skipped..
	using MXKernelType = mcp::kernel::mcp2_maxmin_kernel<T, false, true>;
	MXKernelType mxkernel;  // clamped.
	
	using MXGenType = ::splash::pattern::InnerProduct<MatrixType<T>, MXKernelType, MatrixType<T>, false>;
	using MaskedMXGenType = ::splash::pattern::InnerProduct<MatrixType<T>, MaskedMXKernelType, MatrixType<T>, false>;

	// first max{min}
	MatrixType<T> dmaxmin1(genes_to_genes.rows(), genes_to_genes.columns());
	if (tf_gene_transition == 3) {
		// tf-tf-(tf-tf-gene) and 
		// tf-gen transition on second gene.
		::splash::pattern::InnerProduct<MatrixType<T>, MaskedMXKernelType, MatrixType<T>, true> mxgen;
		mxgen(genes_to_genes, genes_to_genes, maskedmxkernel, dmaxmin1);
	} else {
		::splash::pattern::InnerProduct<MatrixType<T>, MXKernelType, MatrixType<T>, true> mxgen_s;
		// tf-tf-(tf-gene-gene), tf-tf-(gene-gene-gene) tf-gene-(gene-gene-gene), gene-gene-(gene-gene-gene)
		mxgen_s(genes_to_genes, genes_to_genes, mxkernel, dmaxmin1);
	}
	
	auto stime = getSysTime();
	MatrixType<T> maxmin1 = dmaxmin1.allgather();
	auto etime = getSysTime();
	FMT_ROOT_PRINT("Comm,maxmin1 allgather,,{},sec\n", get_duration_s(stime, etime));

	MatrixType<T> dmaxmin2(tfs_to_genes.rows(), genes_to_genes.columns());
	if (tf_gene_transition == 0) { 
		// (tf-gene-gene)-gene-gene on leg 0.
		MXGenType mxgen;
		mxgen(tfs_to_genes, genes_to_genes, mxkernel, dmaxmin2);
	} else {
		MaskedMXGenType mxgen;
		// (tf-tf-tf)-tf-gene  (tf-tf-tf)-gene-gene), (tf-tf-gene)-gene-gene
		mxgen(tfs_to_genes, genes_to_genes, maskedmxkernel, dmaxmin2);
	}
	stime = getSysTime();
	MatrixType<T> maxmin2 = dmaxmin2.allgather();
	etime = getSysTime();
	FMT_ROOT_PRINT("Comm,maxmin2 allgather,,{},sec\n", get_duration_s(stime, etime));


	dmaxmin.resize(tfs_to_genes.rows(), genes_to_genes.columns());
	if ((tf_gene_transition == 2) || (tf_gene_transition == 3)) {  // leges  2, 3
		MaskedMXGenType mxgen;
		// (tf-tf-tf)-tf-gene  (tf-tf-tf)-gene-gene), 
		mxgen(maxmin2, maxmin1, maskedmxkernel, dmaxmin);
	} else { // leg 0
		// (tf-tf-gene)-gene-gene (tf-gene-gene)-gene-gene on leg 0.
		MXGenType mxgen;
		mxgen(maxmin2, maxmin1, mxkernel, dmaxmin);
	}	
}

template <typename T, template <typename> class MatrixType>
void compute_mcp4_tfs(MatrixType<T> const & tfs_to_genes, MatrixType<T> const & genes_to_genes, int const & tf_gene_transition,
	std::vector<T> const & tfs, MatrixType<T> & dmcp) {

	MatrixType<T> dmaxmin;
	compute_maxmin3_tfs(tfs_to_genes, genes_to_genes, tf_gene_transition, tfs, dmaxmin);
	
	compute_ratio(tfs_to_genes, dmaxmin, dmcp);
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


// TODO: compute the dmaxmin1, 2 and 3, dim = tfsXgenes.  assume we are doing ttg, tttg, ttttg, rather than tgg, tggg, ttgg, tgggg, ttggg, tttgg.
// the tgg, tggg, and tgggg cases can be filtered after from ggg, gggg, and ggggg.
// ttgg, ttggg and tttgg cases seem arbitrary.
// an alternative is to do max(tgg, ttg),  max(tggg, ttgg, tttg), max(tgggg, ttggg, tttgg, ttttg)
// thinking : fewer paths to consider (ttg) could lower "max", so expect (ttg < tgg), so max is going to tend towards the tgg, or ggg case.  - this should produce better ratio
// ONLY USEFUL TO EVAL ttg, tttg, ttttg.
template <typename T, template <typename> class MatrixType>
void compute_maxmins_tfs(MatrixType<T> const & tfs_to_genes, 
	MatrixType<T> const & genes_to_genes, 
	std::vector<T> const & tfs, 
	MatrixType<T> & dmaxmin1, 
	MatrixType<T> & dmaxmin2, 
	MatrixType<T> & dmaxmin3 ) {
	
	using MaskedMXKernelType = mcp::kernel::mcp2_maxmin_kernel<T, true, true>;
	MaskedMXKernelType maskedmxkernel(tfs);  // when masking, some columns of first matrix and rows of second are skipped..
	::splash::pattern::InnerProduct<MatrixType<T>, MaskedMXKernelType, MatrixType<T>, false> mxgen;
	::splash::pattern::InnerProduct<MatrixType<T>, MaskedMXKernelType, MatrixType<T>, true> mxgen_s;

	// ** is masked innerproduct.
	// first compute tg ** gg = ttg = dmaxmin1.  
	mxgen(tfs_to_genes, genes_to_genes, maskedmxkernel, dmaxmin1);

	// then ttg ** gg = tttg = dmaxmin2.
	auto stime = getSysTime();	
	MatrixType<T> maxmin1 = dmaxmin1.allgather();
	auto etime = getSysTime();
	FMT_ROOT_PRINT("Comm,maxmin1 allgather,,{},sec\n", get_duration_s(stime, etime));

	mxgen(maxmin1, genes_to_genes, maskedmxkernel, dmaxmin2);

	// compute gg ** gg = gtg
	MatrixType<T> dmaxmin3a;
	mxgen_s(genes_to_genes, genes_to_genes, maskedmxkernel, dmaxmin3a);

	stime = getSysTime();	
	MatrixType<T> maxmin3a = dmaxmin3a.allgather();
	etime = getSysTime();
	FMT_ROOT_PRINT("Comm,maxmin3a allgather,,{},sec\n", get_duration_s(stime, etime));

	// then ttg ** gtg = ttttg = dmaxmin3.
	mxgen(maxmin1, maxmin3a, maskedmxkernel, dmaxmin3);
}



//  combined entire linear combination as a single kernel.  much faster.
template <typename T, template <typename> class MatrixType>
void compute_combo(MatrixType<T> const & mi, 
	MatrixType<T> const & dmaxmin1, 
	MatrixType<T> const & dmaxmin2, 
	MatrixType<T> const & dmaxmin3,
	T const * cs,
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
		double b = cs[1];
		double c = cs[2];
		double d = cs[3];
		double maxmins = 0;

		for (size_t i = 0; i < omp_mxmn_part.size; ++i, ++gid, ++mid) {
			T const * dmi = mi.data(gid);
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

template <typename T, template <typename> class MatrixType>
void compute_stouffer_tf(MatrixType<T> const & dinput, MatrixType<T> & dstouffer) {
	auto stime = getSysTime();
	splash::pattern::Transform<MatrixType<T>, 
		splash::kernel::StandardScore<T, T, false>,
		MatrixType<T>> normalizer;
	splash::kernel::StandardScore<T, T, false> zscore;

	// begin by gathering
	MatrixType<T> tfs_to_genes = dinput.allgather();
	MatrixType<T> genes_to_tfs = tfs_to_genes.local_transpose();

	// these are distributed so need to gather.
	MatrixType<T> tf_norms;
	{
		MatrixType<T> tf_normsD(tfs_to_genes.rows(), tfs_to_genes.columns());
		normalizer(tfs_to_genes, zscore, tf_normsD);
		tf_norms = tf_normsD.allgather();    // not efficient
	}
	MatrixType<T> gene_norms;
	{
		MatrixType<T> gene_normsDT(genes_to_tfs.rows(), genes_to_tfs.columns());
		normalizer(genes_to_tfs, zscore, gene_normsDT);

		// transposed to match tf_norms:  allgather, transpose
		// TODO: make this faster in the aligned_matrix code.
		gene_norms = gene_normsDT.allgather().local_transpose();  // not efficient.
	}

	auto etime = getSysTime();
	FMT_ROOT_PRINT("Zscored in {} sec\n", get_duration_s(stime, etime));
	FMT_PRINT_RT("tf_norms size: {}x{}, gene_norms size: {}x{}\n", tf_norms.rows(), tf_norms.columns(), gene_norms.rows(), gene_norms.columns());

	stime = getSysTime();
	dstouffer.resize(dinput.rows(), tf_norms.columns());
	
	using KernelType = mcp::kernel::zscored_stouffer_kernel<T>;
	KernelType transform;
	splash::pattern::BinaryOp<MatrixType<T>, MatrixType<T>, 
		KernelType,
		MatrixType<T>> transformer;
	// make partitions.
	splash::utils::partition<size_t> input_part = splash::utils::partition<size_t>::make_partition(dinput.rows());
	splash::utils::partition<size_t> output_part(0, dinput.rows(), 0);
	transformer(tf_norms, input_part, gene_norms, input_part, transform, dstouffer, output_part);

	// output should be distributed.
	// do binary op with the 2.
	etime = getSysTime();
	FMT_ROOT_PRINT("transformed, with TF, in {} sec\n", get_duration_s(stime, etime));
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
				auto val2 = std::get<1>(x).compare(std::get<1>(y));
			return (val == 0) ? (val2 < 0) : (val < 0);
	} );
	etime = getSysTime();
	FMT_ROOT_PRINT("sort truth in {} sec\n", get_duration_s(stime, etime));

	return content;
}


// for search.  no longer correlate to row order.  NOT USED
void get_sorted_rowcol_names(
	std::vector<std::string> const & source_gene,
	std::vector<std::string> const & target_gene,
	std::vector<std::pair<std::string, size_t>> & src_names,
	std::vector<std::pair<std::string, size_t>> & dst_names) {

	// generate sorted row and col names, with id to original array.
	auto comp = [](std::pair<std::string, size_t> const & x, std::pair<std::string, size_t> const & y) {
		return x.first.compare(y.first) < 0;
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
void get_sorted_local_rowcol_names(
	size_t const & local_rows,
	std::vector<std::string> const & source_gene,
	std::vector<std::string> const & target_gene,
	std::vector<std::pair<std::string, size_t>> & dsrc_names,
	std::vector<std::pair<std::string, size_t>> & dst_names) {

	// generate sorted row and col names, with id to original array.
	auto comp = [](std::pair<std::string, size_t> const & x, std::pair<std::string, size_t> const & y) {
		return x.first.compare(y.first) < 0;
	};

	splash::utils::partition<size_t> mpi_part = splash::utils::partition<size_t>::make_partition(local_rows);

	dsrc_names.clear();
	dsrc_names.reserve(mpi_part.size);
	size_t id = mpi_part.offset;
	for (size_t i = 0; i < mpi_part.size; ++i, ++id) {
		dsrc_names.emplace_back(source_gene[id], i);  // dsrc_names is either for full or tf, with row numbers being local.
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
// NOTE:  output will start with (1, 0). We are only working with lower triangle, and diagonal does not count. 
//     out put will end with (c, c-1)
// truth is full list, and src_names is local.
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



std::vector<std::tuple<size_t, size_t, int>> 
select_from_list(size_t const & local_rows,  // partitioning of the genes (tf, or full)
	std::vector<std::tuple<std::string, std::string, int>> const & truth,  // truth list - tf or full
	std::vector<std::string> const & source_gene,
	std::vector<std::string> const & target_gene
){
	std::vector<std::pair<std::string, size_t>> dsrc_names;
	std::vector<std::pair<std::string, size_t>> dst_names;

	get_sorted_local_rowcol_names(
			local_rows, 
			source_gene, target_gene,
			dsrc_names, dst_names);

	// FMT_PRINT_RT("[DEBUG] get mask: truth {}, local row {}, src {} x {}, out {} x {}\n", 
	// 	truth.size(), local_rows, source_gene.size(), target_gene.size(), dsrc_names.size(), dst_names.size());

	return select_mask(truth, dsrc_names, dst_names);
}


// upper triangle only (should be same as lower triangle).  in global coords.
std::vector<std::tuple<size_t, size_t, int>> 
select_lower_triangle(
	size_t const & local_rows,  // partitioning of the full genes
	splash::ds::aligned_matrix<char> const & full_mat) {

	auto stime = getSysTime();
	splash::utils::partition<size_t> mpi_part = splash::utils::partition<size_t>::make_partition(local_rows);

	std::vector<std::tuple<size_t, size_t, int>> mask;
	mask.reserve((local_rows * full_mat.columns()) >> 1);

	char v;
	const char * vptr;
	size_t rid = mpi_part.offset;
	size_t last;
	for (size_t i = 0; i < mpi_part.size; ++i, ++rid) {
		vptr = full_mat.data(rid);
		last = std::min(full_mat.columns(), rid);
		for (size_t j = 0; j < last; ++j, ++vptr) {
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

// upper triangle only (should be same as lower triangle).  in global coords of the tf matrix.
template<typename IT>
std::vector<std::tuple<size_t, size_t, int>> 
select_tfs_from_mat(  // 
	size_t const & local_rows, std::vector<IT> const & tfs, 
	splash::ds::aligned_matrix<char> const & tf_mat) {

		// need to keep all TF-GENE, regardless of where in the matrix triangle.  
		// include TF1-TF2, but exclude TF2-TF1, and TF1-TF1.

	// the local_rows correspond to partions in the TFs array after filtering out nonTFs.
	auto stime = getSysTime();
	splash::utils::partition<size_t> tf_mat_part = splash::utils::partition<size_t>::make_partition(local_rows);

	// make an id list.  this is what will be partitioned by processor.
	std::vector<size_t> tf_rowid_in_full;
	for (size_t i = 0; i < tfs.size(); ++i) {
		if (tfs[i] > 0.0) tf_rowid_in_full.emplace_back(i);
	}

	std::vector<std::tuple<size_t, size_t, int>> mask;
	mask.reserve(local_rows * tf_mat.columns());

	char v;
	const char * vptr;
	size_t lrid = tf_mat_part.offset; // local row_id, not global location.
	size_t grid;  // global rid get from the tf_rowid_in_full array.
	size_t last = tf_mat.columns();
	for (size_t i = 0; i < tf_mat_part.size; ++i, ++lrid) {

		vptr = tf_mat.data(lrid);
		grid = tf_rowid_in_full[lrid];
		for (size_t j = 0; j < last; ++j, ++vptr) {
			if (grid == j) continue; // diagonal in global coord.
			if ((grid > j) && (tfs[j] > 0))  continue;  // lower triangle AND target is a TF, so already included, don't add again.

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


// walking through appears to work properly.  negative probs all come before positive probes.
// aupr code is validated to work correctly
// mask:  sorted.  (once)  could be distributed.
// value: could be distribued.

template<typename T>
void select_values2(std::vector<std::tuple<size_t, size_t, int>>::const_iterator mask_start,
	std::vector<std::tuple<size_t, size_t, int>>::const_iterator mask_end,
	splash::ds::aligned_matrix<T> const & dvalues, 
	splash::ds::aligned_vector<T> & pos, splash::ds::aligned_vector<T> & neg) {

	// now search rows, followed by column.
	auto stime = getSysTime();
	size_t npos = 0, nneg = 0;
	int v;
	for (auto it = mask_start; it != mask_end; ++it) {
		v = std::get<2>(*it);
		npos += (v == 1);
		nneg += (v == 0);
	}

	pos.resize(npos);
	neg.resize(nneg);
	auto etime = getSysTime();
	FMT_ROOT_PRINT("count.  pos {} neg {} in {} sec\n", npos, nneg, get_duration_s(stime, etime));

	// stime = getSysTime();	
	// splash::ds::aligned_matrix<double> values = dvalues.allgather();
	// etime = getSysTime();
	// FMT_ROOT_PRINT("Comm,aupr values allgather,,{},sec\n", get_duration_s(stime, etime));


	stime = getSysTime();
	// walk through the mask
	size_t i = 0, j = 0;
	size_t r, c;
	for (auto mask_i = mask_start; mask_i != mask_end; ++mask_i) {
		r = std::get<0>(*mask_i);
		c = std::get<1>(*mask_i);
		v = std::get<2>(*mask_i);
		if (v == 0) {
			// FMT_ROOT_PRINT("getting pos value at [{} {}] to neg {} \n", std::get<0>(m), std::get<1>(m), j);
			neg.at(j) = dvalues.at(r, c);
			++j;
		} else if (v == 1) {
			// FMT_ROOT_PRINT("getting pos value at [{} {}] to pos {} \n", std::get<0>(m), std::get<1>(m), i);
			pos.at(i) = dvalues.at(r, c);
			++i;
		}
	}
	pos.resize(i);
	neg.resize(j);
	FMT_ROOT_PRINT("pos size {}, neg size {}", pos.size(), neg.size());
	
	etime = getSysTime();
	FMT_ROOT_PRINT("extracted values.  pos {} neg {} in {} sec\n", pos.size(), neg.size(), get_duration_s(stime, etime));
}

template<typename T>
void select_values(std::vector<std::tuple<size_t, size_t, int>> const & mask,
	splash::ds::aligned_matrix<T> const & dvalues, 
	splash::ds::aligned_vector<T> & pos, splash::ds::aligned_vector<T> & neg) {

	select_values2(mask.cbegin(), mask.cend(), dvalues, pos, neg);

}




// training vs testing:  we are just trying to optimize based on a single data point
// better quesiton - does optimized based on partial data produce better overall output
//  3 subquestions - 1. what are the value ranges if we were to split the ground truth, compared to the full ground truth?  (non-combo, permute)
//		2. if we were to split differently?  (non-combo, change proportion)
// 		3. if we were to select combo based on partial ground truth, does the whole network do better?  (combo, permute truth)
// 	
template <typename Kernel, typename T = double, typename O = double>
O compute_aupr2(std::vector<std::tuple<size_t, size_t, int>>::const_iterator mask_start,
	std::vector<std::tuple<size_t, size_t, int>>::const_iterator mask_end,
	splash::ds::aligned_matrix<T> const & dvals, 
	Kernel const & auprkern,
	splash::ds::aligned_vector<double> & pos, splash::ds::aligned_vector<double> & neg) {

	// compute AUPR for the full dataset.
	auto stime = getSysTime();
	select_values2(mask_start, mask_end, dvals, pos, neg);
	auto etime = getSysTime();
	FMT_ROOT_PRINT("Select,truth,,{},sec\n", get_duration_s(stime, etime));

	stime = getSysTime();
	O aupr = auprkern(pos, neg);
	etime = getSysTime();
	FMT_ROOT_PRINT("Computed,aupr {},,{},sec\n", aupr, get_duration_s(stime, etime));

	return aupr;
}
template <typename Kernel, typename T = double, typename O = double>
O compute_aupr(std::vector<std::tuple<size_t, size_t, int>> const & dmask,
	splash::ds::aligned_matrix<T> const & dvals, 
	Kernel const & auprkern,
	splash::ds::aligned_vector<double> & pos, splash::ds::aligned_vector<double> & neg) {

	return compute_aupr2(dmask.cbegin(), dmask.cend(), dvals, auprkern, pos, neg);
}




template <typename Kernel, typename Kernel2, typename T, typename O = T, typename L = char>
O compute_aupr_auroc(std::vector<std::tuple<size_t, size_t, int>> const & dmask,
	splash::ds::aligned_matrix<T> const & dvals, 
	Kernel const & auprkern, Kernel2 const & aurockern,
	splash::ds::aligned_vector<T> & pos, splash::ds::aligned_vector<T> & neg) {

	auto stime = getSysTime();
	select_values(dmask, dvals, pos, neg);
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

template<typename IT>
void load_tfs(std::string const & tf_input, std::vector<std::string> const & genes, 
	std::vector<IT> & tfs, std::vector<std::string> & tf_names,
	int const & rank = 0) {

		if (tf_input.length() == 0 ) return;

	std::unordered_set<std::string> TFs;  // no need to return
	std::unordered_set<std::string> gns(genes.begin(), genes.end());  // no need to return


	auto stime = getSysTime();

	// read the file and put into a set.
	std::ifstream ifs(tf_input);
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
		tfs.push_back(res); // if this is one of the target TF, then save it
		if (res >= 0.0) {
			tf_names.push_back(genes[i]);
		}
	}

	auto etime = getSysTime();
	FMT_ROOT_PRINT("Load TF data in {} sec\n", get_duration_s(stime, etime));
	if (rank == 0) {
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
}

template<typename MatrixType, typename IT>
void filter_mat_rows_by_tfs(MatrixType const & input, 
	std::vector<IT> const & tfs,
	MatrixType & row_selected) {

	size_t tf_count = 0;
	for (size_t i = 0; i < tfs.size(); ++i) {
		tf_count += (tfs[i] > 0);
	}

	//======== NOT making smaller arrays.
	// select rows.
	row_selected.resize(tf_count, input.columns());
	for (size_t i = 0, j = 0; i < input.rows(); ++i) {
		if (tfs[i] > 0) {
			memcpy(row_selected.data(j), input.data(i), input.column_bytes());
			++j;
		}
	}
}

template<typename MatrixType, typename IT>
void filter_local_mat_rows_by_tfs(MatrixType const & input, 
	std::vector<IT> const & tfs,
	MatrixType & row_selected) {

	splash::utils::partition<size_t> mpi_part = splash::utils::partition<size_t>::make_partition(input.rows());
	size_t tf_count = 0;

	auto rid = mpi_part.offset;
	for (size_t i = 0; i < mpi_part.size; ++i, ++rid) {
		tf_count += (tfs[rid] > 0);
	}

	//======== NOT making smaller arrays.
	// select rows.
	row_selected.resize(tf_count, input.columns());
	rid = mpi_part.offset;
	for (size_t i = 0, j = 0; i < mpi_part.size; ++i, ++rid) {
		if (tfs[rid] > 0) {
			memcpy(row_selected.data(j), input.data(i), input.column_bytes());
			++j;
		}
	}
}


// only need to select the src (row), not the target (column)
void filter_list_by_tfs(std::vector<std::tuple<std::string, std::string, int>> const & input, 
	std::vector<std::string> const & tf_names,
	std::vector<std::tuple<std::string, std::string, int>> & row_selected) {

	if (tf_names.size() == 0) return;

	// copy and sort.
	std::vector<std::string> sorted(tf_names.begin(), tf_names.end());
	std::sort(sorted.begin(), sorted.end(), [](std::string const & x, std::string const & y){
		return x.compare(y) < 0;
	});

	// iterate over all input, search by name in sorted names.  if present, insert into new.
	row_selected.clear();
	for (auto entry : input) {
		std::string src = std::get<0>(entry);
		if (std::binary_search(sorted.begin(), sorted.end(), src)) {
			row_selected.emplace_back(entry);
		}
	}
}
