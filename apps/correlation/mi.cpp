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
		enum method_type : int { BSPLINE = 0, ADAPTIVE_FAST = 1 };


		int num_bins;
		int spline_order;
		method_type method;

		app_parameters() : num_bins(10), spline_order(3), method(ADAPTIVE_FAST) {}
		virtual ~app_parameters() {}

		virtual void config(CLI::App& app) {
            app.add_option("-b,--bins", num_bins, "No. of bins")->group("MI")->capture_default_str();
			app.add_option("-l,--spline_order", spline_order, "Spline Order")->group("MI")->capture_default_str();	
			app.add_option("-m,--method", method, "Algorithm: B-spline MI=0, Adaptive partitioning=1")->group("MI")->capture_default_str();
		}

		virtual void print(const char * prefix) const {
            FMT_ROOT_PRINT("{} MI Parameters       : \n", prefix);
            FMT_ROOT_PRINT("{} ->Number of bins    : {}\n", prefix, num_bins); 
            FMT_ROOT_PRINT("{} ->Spline Order      : {}\n", prefix, spline_order); 
			FMT_ROOT_PRINT("{} ->MI compute method : {}\n", prefix, 
				(method == BSPLINE ? "b-spline" : "Adaptive Partitioning" ));
		}
};

template<typename DataType>
void run(splash::io::common_parameters& common_params,
         splash::io::mpi_parameters& mpi_params,
         app_parameters& app_params ){
	// =============== SETUP INPUT ===================
	// NOTE: input data is replicated on all MPI procs.
	splash::ds::aligned_matrix<DataType> input;
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
	etime = getSysTime();
	FMT_ROOT_PRINT("Load data in {} sec\n", get_duration_s(stime, etime));

	if (mpi_params.rank == 0) {
		mpi_params.print("[PARAM] ");
		common_params.print("[PARAM] ");
		app_params.print("[PARAM] ");
        std::cout << "No. GENES    : " << genes.size() << std::endl;
        std::cout << "No. SAMPLES  : " << samples.size() << std::endl;
        std::cout << "INPUT SIZE   : " << input.rows() << "x" << input.columns() << std::endl;
	}
	if (app_params.method == app_parameters::method_type::ADAPTIVE_FAST) {

		// 1. Rank transform
		// ------------ add noise -----------
		// this is a critical step.
		// ------------ perturb -----------
		splash::ds::aligned_matrix<DataType> noisy(input.rows(), input.columns());
		auto stime = getSysTime();
		// ---- create a VV2S kernel
		using NoiseKernelType = ::mcp::kernel::add_noise<DataType, std::uniform_real_distribution<DataType>, std::default_random_engine>;
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
		using RankKernelType = splash::kernel::Rank<DataType, size_t, 0, false>;   // descending
		splash::pattern::Transform<splash::ds::aligned_matrix<DataType>, 
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
		using MIKernelType = mcp::kernel::AdaptivePartitionRankMIKernel2<size_t, DataType, 0>;
		splash::pattern::InnerProduct<
			splash::ds::aligned_matrix<size_t>, 
			MIKernelType,
			splash::ds::aligned_matrix<DataType>, true > adaptive_mi;
		MIKernelType mi_op(histos.data(), histos.column_bytes());
		splash::ds::aligned_matrix<DataType> output(ranked.rows(), ranked.rows());
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
		splash::pattern::GlobalTransform<splash::ds::aligned_matrix<DataType>, 
			mcp::kernel::MinMaxScale<DataType>,
			splash::ds::aligned_matrix<DataType>> scaler;
		mcp::kernel::MinMaxScale<DataType> scaling_op;
		splash::ds::aligned_matrix<DataType> dscaled;
		scaler(input, scaling_op, dscaled);
		splash::ds::aligned_matrix<DataType> scaled = dscaled.allgather();

		etime = getSysTime();
		FMT_ROOT_PRINT("Scaled Partitioned in {} sec\n", get_duration_s(stime, etime));

		// 
		// 2. Compute Weights : Vector -> Weight Vector
		// 
		stime = getSysTime();
		int num_samples = (int) input.columns();
		splash::pattern::Transform<splash::ds::aligned_matrix<DataType>, 
			mcp::kernel::BSplineWeightsKernel<DataType>,
			splash::ds::aligned_matrix<DataType>> bspline_weights;
		splash::ds::aligned_matrix<DataType> dweighted(dscaled.rows(), 
			dscaled.columns() * app_params.num_bins + 1);
		mcp::kernel::BSplineWeightsKernel<DataType> weighting_op(app_params.num_bins,
			app_params.spline_order, num_samples);
		bspline_weights(dscaled, weighting_op, dweighted);	
		splash::ds::aligned_matrix<DataType> weighted = dweighted.allgather();
		etime = getSysTime();
		FMT_ROOT_PRINT("Weight Partitioned in {} sec\n", get_duration_s(stime, etime));

		// //
		// // 3. Entropy : Weight Vector -> Entropy
		// //
		// stime = getSysTime();
		// splash::ds::aligned_vector<DataType> summarized(input.rows());
		// splash::pattern::Reduce<splash::ds::aligned_matrix<DataType>, 
		// 	mcp::kernel::Entropy1DKernel<DataType, DataType>,
		// 	splash::ds::aligned_vector<DataType>, splash::pattern::DIM_INDEX::ROW> wts_entropy1d;
		// mcp::kernel::Entropy1DKernel<DataType, DataType> entropy1d;
		// wts_entropy1d(weighted, entropy1d, summarized); 

		// etime = getSysTime();
		// FMT_PRINT_MPI_ROOT("Weighted Partitioned in {} sec\n", get_duration_s(stime, etime));


		// 4. Inner Product :  Weight Vector, Weight Vector -> MI
		stime = getSysTime();
		// ---- create a VV2S kernel
		splash::pattern::InnerProduct<
			splash::ds::aligned_matrix<DataType>, 
			mcp::kernel::BSplineMIKernel<DataType>,
			splash::ds::aligned_matrix<DataType>, true > mi_proc;
		// mcp::kernel::BSplineMIKernel<DataType> mi_op(app_params.num_bins,
	 //										num_samples);
		splash::ds::aligned_matrix<DataType> output(input.rows(), input.rows());
		// mi_proc(weighted, weighted, mi_op, output);

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
}

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

    if(common_params.use_single) {
        run<float>(common_params, mpi_params, app_params);
    } else {
        run<double>(common_params, mpi_params, app_params);
    }

}
