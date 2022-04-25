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
#include <string>
#include <vector>

#include "CLI/CLI.hpp"
#include "splash/io/CLIParserCommon.hpp"
#include "splash/io/parameters_base.hpp"
#include "splash/utils/benchmark.hpp"
#include "splash/utils/report.hpp"
#include "splash/ds/aligned_matrix.hpp"
#include "splash/patterns/pattern.hpp"
#include "mcp/transform/combine.hpp"


#include "splash/io/matrix_io.hpp"


#ifdef USE_OPENMP
#include <omp.h>
#endif

class app_parameters : public parameters_base {
	protected:
		std::vector<std::string> method_names = {"MIN", "MAX", "ADD", "MADD", "SUB", "MULT", "DIV"};

	public:
		enum method_type : int { MIN = 1, MAX = 2, ADD = 3, MADD = 4, SUB = 5, MULT = 6, DIV = 7 };

		method_type method;
		std::vector<std::string> inputs;
        std::vector<double> coeffs;
		std::string output;
        size_t num_threads;

		app_parameters() : method(MADD), num_threads(1) {}
		virtual ~app_parameters() {}

		virtual void config(CLI::App& app) {
            app.add_option("-m,--method", method, "Algorithm: MIN=1, MAX = 2, ADD = 3, MADD = 4")->group("Method");

			// allow input, or random input, or piped input later. 
            app.add_option("-i,--input", inputs, "input files (1 or more)")->required()->each(CLI::ExistingFile)->group("Input");
			app.add_option("-c,--coefficient", coeffs, "coefficients (1 or more)")->group("Input");

            // output
            app.add_option("-o,--output", output, "output file")->group("Output");

            // default to 1 thread.
            app.add_option("-t,--threads", num_threads, "number of CPU threads")->group("Hardware")->check(CLI::PositiveNumber);

            // ensure that number of inputs is <= number of coefficients.

        }

        virtual void print(const char* prefix) {
            FMT_ROOT_PRINT("{} combination method: {}\n", prefix, method_names[method - 1]); 
            // FMT_ROOT_PRINT("Single precision: {}\n", use_single ? 1 : 0);
			for (auto input : inputs) 
	            FMT_ROOT_PRINT("{} Input: {}\n", prefix, input.c_str());
			for (auto c : coeffs)
	            FMT_ROOT_PRINT("{} Input: {}\n", prefix, c);
	
			FMT_ROOT_PRINT("{} Output: {}\n", prefix, output.c_str());
            FMT_ROOT_PRINT("{} Number of threads: {}\n", prefix, num_threads);

		}
};



int main(int argc, char* argv[]) {

	//==============  PARSE INPUT =====================
	CLI::App app{"Correlation Transform"};

	// handle MPI (TODO: replace with MXX later)
	splash::io::mpi_parameters mpi_params(argc, argv);
	mpi_params.config(app);

	// set up CLI parsers.
	app_parameters app_params;
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
	omp_set_num_threads(app_params.num_threads);
#endif

	// rely on default double constructor to produce 0.0
	app_params.coeffs.resize(app_params.inputs.size());

	// =============== SETUP INPUT ===================
	// NOTE: input data is replicated on all MPI procs.
	using MatrixType = splash::ds::aligned_matrix<double>;
	
	MatrixType first, next;
	size_t first_rows = 0, next_rows = 0;
	size_t first_cols = 0, next_cols = 0;
	std::vector<std::string> first_genes, next_genes;
	std::vector<std::string> first_samples, next_samples;
	
	auto stime = getSysTime();
	auto etime = getSysTime();

	stime = getSysTime();
	MatrixType inmat = read_matrix<double>(app_params.inputs[0], 
		"array",
		first_rows, first_cols,
		first_genes, first_samples);
	first = inmat.scatter();

	FMT_PRINT_RT("read first matrix.  size: {}x{}, names {}x{}, actually data {}x{}\n", first_rows, first_cols, first_genes.size(), first_samples.size(), first.rows(), first.columns());

	MatrixType output(first.rows(), first.columns());

	// scale input and set as output.
	if (app_params.method == app_parameters::method_type::MADD) {
		using ScaleKernelType = mcp::kernel::scale_kernel<double>;
		using ScaleGenType = ::splash::pattern::Transform<MatrixType, ScaleKernelType, MatrixType>;
		ScaleKernelType scaler(app_params.coeffs[0]);
		ScaleGenType scalegen;

		if (app_params.coeffs[0] != 0.0) {
			scalegen(first, scaler, output);
			first = output;
		} else {
			first.zero();
		}
	}
	etime = getSysTime();
	FMT_ROOT_PRINT("Load first data in {} sec\n", get_duration_s(stime, etime));


	if (mpi_params.rank == 0) {
		mpi_params.print("[PARAM] ");
		app_params.print("[PARAM] ");
	}



	// now combine.
	stime = getSysTime();
	using AddKernelType = mcp::kernel::add_kernel<double>;
	using AddGenType = ::splash::pattern::BinaryOp<MatrixType, MatrixType, AddKernelType, MatrixType>;
	AddKernelType addkernel; 
	AddGenType addgen;

	using SubKernelType = mcp::kernel::sub_kernel<double>;
	using SubGenType = ::splash::pattern::BinaryOp<MatrixType, MatrixType, SubKernelType, MatrixType>;
	SubKernelType subkernel; 
	SubGenType subgen;

	using MultiplyKernelType = mcp::kernel::multiply_kernel<double>;
	using MultiplyGenType = ::splash::pattern::BinaryOp<MatrixType, MatrixType, MultiplyKernelType, MatrixType>;
	MultiplyKernelType multiplykernel; 
	MultiplyGenType multiplygen;

	using DivideKernelType = mcp::kernel::ratio_kernel<double>;
	using DivideGenType = ::splash::pattern::BinaryOp<MatrixType, MatrixType, DivideKernelType, MatrixType>;
	DivideKernelType divkernel; 
	DivideGenType divgen;


	using AvgKernelType = mcp::kernel::madd_kernel<double>;
	using AvgGenType = ::splash::pattern::BinaryOp<MatrixType, MatrixType, AvgKernelType, MatrixType>;
	AvgGenType avggen;

	using MaxKernelType = mcp::kernel::max_kernel<double>;
	using MaxGenType = ::splash::pattern::BinaryOp<MatrixType, MatrixType, MaxKernelType, MatrixType>;
	MaxKernelType maxkernel;
	MaxGenType maxgen;

	using MinKernelType = mcp::kernel::min_kernel<double>;
	using MinGenType = ::splash::pattern::BinaryOp<MatrixType, MatrixType, MinKernelType, MatrixType>;
	MinKernelType minkernel;
	MinGenType mingen;

	next_rows = 0;
	next_cols = 0;

	for (size_t i = 1; i < app_params.inputs.size(); ++i) {
		next = read_matrix<double>(app_params.inputs[i], 
			"array",
			next_rows, next_cols,
			next_genes, next_samples).scatter();

		if ((next_rows != first_rows) || 
			(next_cols != first_cols)) continue;


		if (app_params.method == app_parameters::method_type::MADD) {
			if (app_params.coeffs[i] != 0.0) {
				AvgKernelType avgkernel(app_params.coeffs[i]); 
				avggen(first, next, avgkernel, output);
			}
		} else if (app_params.method == app_parameters::method_type::ADD) {
			addgen(first, next, addkernel, output);
		} else if (app_params.method == app_parameters::method_type::SUB) {
			subgen(first, next, subkernel, output);
		} else if (app_params.method == app_parameters::method_type::MULT) {
			multiplygen(first, next, multiplykernel, output);
		} else if (app_params.method == app_parameters::method_type::DIV) {
			divgen(first, next, divkernel, output);
		} else if (app_params.method == app_parameters::method_type::MAX) {
			maxgen(first, next, maxkernel, output);
		} else if (app_params.method == app_parameters::method_type::MIN) {
			mingen(first, next, minkernel, output);
		} else {
			FMT_PRINT_ERR("ERROR: unsupported method");
		}

		first = output;
	}
	etime = getSysTime();
	FMT_ROOT_PRINT("Load rest of data and combine in {} sec\n", get_duration_s(stime, etime));

	// =============== WRITE OUT ==============
	// NOTE: rank 0 writes out.
	stime = getSysTime();
	write_matrix_distributed(app_params.output, "array", first_genes, first_samples, output);
	etime = getSysTime();
	FMT_ROOT_PRINT("Output in {} sec\n", get_duration_s(stime, etime));
	FMT_FLUSH();

	return 0;
}
