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
#include "splash/io/EXPMatrixWriter.hpp"
#include "splash/utils/benchmark.hpp"
#include "splash/utils/report.hpp"
#include "splash/ds/aligned_matrix.hpp"
#include "splash/kernel/dotproduct.hpp"
#include "splash/patterns/pattern.hpp"

#include "splash/io/matrix_io.hpp"
#include "mcp/correlation/BSplineMI.hpp"
#include "mcp/correlation/AdaptivePartitioningMI.hpp"
#include "splash/transform/rank.hpp"

#ifdef USE_OPENMP
#include <omp.h>
#endif


class app_parameters : public parameters_base {
	public:
		enum method_type : int { BSPLINE = 1, ADAPTIVE_FAST = 2 };


		int num_bins;
		int spline_order;
		method_type method;

		app_parameters() : num_bins(10), spline_order(3), method(ADAPTIVE_FAST) {}
		virtual ~app_parameters() {}

		virtual void config(CLI::App& app) {
            app.add_option("-b,--bins", num_bins, "No. of bins")->group("MI");
			app.add_option("-s,--spline_order", spline_order, "Spline Order")->group("MI");	
			app.add_option("-m,--method", method, "Algorithm: B-spline MI=1, Adaptive partitioning=2")->group("MI");
		}

		virtual void print(const char * prefix) {
            FMT_ROOT_PRINT("{} Number of bins: [{}] Spline Order: [{}]\n", prefix,
			      num_bins, spline_order); 
			FMT_ROOT_PRINT("{} MI compute method: {}\n", prefix, 
				(method == BSPLINE ? "b-spline" : "Adaptive Partitioning" ));
		}
};

int main(int argc, char* argv[]) {

	//==============  PARSE INPUT =====================
	CLI::App app{"MI computation"};

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
		input = read_matrix<double>(common_params.input, "array", 
			common_params.num_vectors, common_params.vector_size,
			genes, samples, common_params.skip );
	}
	etime = getSysTime();
	FMT_ROOT_PRINT("Load data in {} sec\n", get_duration_s(stime, etime));

	if (mpi_params.rank == 0) {
		mpi_params.print("[PARAM] ");
		common_params.print("[PARAM] ");
		app_params.print("[PARAM] ");
	}
	if (app_params.method == app_parameters::method_type::ADAPTIVE_FAST) {

		// 1. Rank transform
		// ------------ add noise -----------
		// this is a critical step.
		// ------------ perturb -----------
		splash::ds::aligned_matrix<double> noisy(input.rows(), input.columns());
		auto stime = getSysTime();
		// ---- create a VV2S kernel
		using NoiseKernelType = ::wave::kernel::add_noise<double, std::uniform_real_distribution<double>, std::default_random_engine>;
		splash::kernel::random_number_generator<> gen;
		NoiseKernelType noise_adder(gen, 0.0, 0.00001);
		noisy.resize(input.rows(), input.columns()); 
		noise_adder(input, noisy);

		auto etime = getSysTime();
		FMT_ROOT_PRINT("Noise added in {} sec\n", get_duration_s(stime, etime));

		//
		// ------------ normalize -----------  looks okay
		stime = getSysTime();
		// ---- create a VV2S kernel
		using RankKernelType = splash::kernel::Rank<double, size_t, 0, false>;   // descending
		splash::pattern::Transform<splash::ds::aligned_matrix<double>, 
			RankKernelType,
			splash::ds::aligned_matrix<size_t>> normalizer;
		RankKernelType ranker;
		splash::ds::aligned_matrix<size_t> ranked(noisy.rows(), noisy.columns()); 
		normalizer(noisy, ranker, ranked);

		etime = getSysTime();
		FMT_ROOT_PRINT("Rank Transformed in {} sec\n", get_duration_s(stime, etime));

		// 2. histogram
		// ------------ scale -----------  looks okay
		stime = getSysTime();
		// ---- create a VV2S kernel
		using HistogramKernelType = wave::kernel::IntegralCumulativeHistogramKernel<size_t, 0>;
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
		using MIKernelType = wave::kernel::AdaptivePartitionRankMIKernel2<size_t, double, 0>;
		splash::pattern::InnerProduct<
			splash::ds::aligned_matrix<size_t>, 
			MIKernelType,
			splash::ds::aligned_matrix<double>, true > adaptive_mi;
		MIKernelType mi_op(histos.data(), histos.column_bytes());
		splash::ds::aligned_matrix<double> output(ranked.rows(), ranked.rows());
		adaptive_mi(ranked, ranked, mi_op, output);

		etime = getSysTime();
		FMT_ROOT_PRINT("Compute Adpative MI in {} sec\n", get_duration_s(stime, etime));

		// FMT_ROOT_PRINT("MI [0, 0:4] {} {} {} {} {}\n", output.at(0,0), output.at(0,1), output.at(0,2), output.at(0,3), output.at(0,4))

		// =============== WRITE OUT ==============
		// NOTE: rank 0 writes out.
		stime = getSysTime();
		write_matrix_distributed(common_params.output, "array", genes, genes, output);
		etime = getSysTime();
		FMT_ROOT_PRINT("Output in {} sec\n", get_duration_s(stime, etime));
		FMT_FLUSH();

	} else {

		// 1. Scale : Vector -> Scaled Vector
		// ------------ scale -----------
		//
		stime = getSysTime();
		splash::pattern::GlobalTransform<splash::ds::aligned_matrix<double>, 
			wave::kernel::MinMaxScale<double>,
			splash::ds::aligned_matrix<double>> scaler;
		wave::kernel::MinMaxScale<double> scaling_op;
		splash::ds::aligned_matrix<double> dscaled;
		scaler(input, scaling_op, dscaled);
		splash::ds::aligned_matrix<double> scaled = dscaled.allgather();

		etime = getSysTime();
		FMT_ROOT_PRINT("Scaled Partitioned in {} sec\n", get_duration_s(stime, etime));

		// 
		// 2. Compute Weights : Vector -> Weight Vector
		// 
		stime = getSysTime();
		int num_samples = (int) input.columns();
		splash::pattern::Transform<splash::ds::aligned_matrix<double>, 
			wave::kernel::BSplineWeightsKernel<double>,
			splash::ds::aligned_matrix<double>> bspline_weights;
		splash::ds::aligned_matrix<double> dweighted(dscaled.rows(), 
			dscaled.columns() * app_params.num_bins + 1);
		wave::kernel::BSplineWeightsKernel<double> weighting_op(app_params.num_bins,
			app_params.spline_order, num_samples);
		bspline_weights(dscaled, weighting_op, dweighted);	
		splash::ds::aligned_matrix<double> weighted = dweighted.allgather();
		etime = getSysTime();
		FMT_ROOT_PRINT("Weight Partitioned in {} sec\n", get_duration_s(stime, etime));

		// //
		// // 3. Entropy : Weight Vector -> Entropy
		// //
		// stime = getSysTime();
		// splash::ds::aligned_vector<double> summarized(input.rows());
		// splash::pattern::Reduce<splash::ds::aligned_matrix<double>, 
		// 	wave::kernel::Entropy1DKernel<double, double>,
		// 	splash::ds::aligned_vector<double>, splash::pattern::DIM_INDEX::ROW> wts_entropy1d;
		// wave::kernel::Entropy1DKernel<double, double> entropy1d;
		// wts_entropy1d(weighted, entropy1d, summarized); 

		// etime = getSysTime();
		// FMT_PRINT_MPI_ROOT("Weighted Partitioned in {} sec\n", get_duration_s(stime, etime));


		// 4. Inner Product :  Weight Vector, Weight Vector -> MI
		stime = getSysTime();
		// ---- create a VV2S kernel
		splash::pattern::InnerProduct<
			splash::ds::aligned_matrix<double>, 
			wave::kernel::BSplineMIKernel<double>,
			splash::ds::aligned_matrix<double>, true > bspline_mi;
		wave::kernel::BSplineMIKernel<double> mi_op(app_params.num_bins,
											num_samples);
		splash::ds::aligned_matrix<double> output(input.rows(), input.rows());
		bspline_mi(weighted, weighted, mi_op, output);

		etime = getSysTime();
		FMT_ROOT_PRINT("Compute MI in {} sec\n", get_duration_s(stime, etime));

		// =============== WRITE OUT ==============
		// NOTE: rank 0 writes out.
		stime = getSysTime();
		write_matrix_distributed(common_params.output, "array", genes, genes, output);
		etime = getSysTime();
		FMT_ROOT_PRINT("Output in {} sec\n", get_duration_s(stime, etime));
		FMT_FLUSH();

	}

	return 0;

}

