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

#include "splash/io/matrix_io.hpp"


#ifdef USE_OPENMP
#include <omp.h>
#endif



class app_parameters : public parameters_base {
	public:

        std::string row_names;
        std::string col_names;

		app_parameters() {}
		virtual ~app_parameters() {}

		virtual void config(CLI::App& app) {
			app.add_option("-r,--rows", row_names, "names of rows to keep, 1 per line.")->group("selection");
			app.add_option("-c,--columns", col_names, "names of columns to keep, 1 per line.")->group("selection");
		}
		virtual void print(const char * prefix) const {
			FMT_ROOT_PRINT("{} row names file         : {}\n", prefix, row_names.c_str()); 
            FMT_ROOT_PRINT("{} column names file      : {}\n", prefix, col_names.c_str()); 
            
		}
        
};

template<typename DataType>
void run(splash::io::common_parameters& common_params,
         splash::io::mpi_parameters& mpi_params, app_parameters& app_params) {

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
		input = read_matrix<DataType>(common_params.input, common_params.h5_group, 
			common_params.num_vectors, common_params.vector_size,
			genes, samples, common_params.skip, 1, common_params.h5_gene_key,
            common_params.h5_samples_key, common_params.h5_matrix_key);
	}
	etime = getSysTime();
	FMT_ROOT_PRINT("Load data in {} sec\n", get_duration_s(stime, etime));


	
	if (mpi_params.rank == 0) {
		mpi_params.print("[PARAM] ");
		common_params.print("[PARAM] ");
		app_params.print("[PARAM] ");
	}

	// =============== PARTITION and RUN ===================
	// FMT_PRINT("genes {}, samples {}\n", genes.size(), samples.size());

	std::unordered_set<std::string> gns(genes.begin(), genes.end());
	std::unordered_set<std::string> smpls(samples.begin(), samples.end());
	std::vector<bool> row_selected, col_selected;
	std::vector<std::string> r_names, c_names;
	bool select_rows = app_params.row_names.length() > 0;
	bool select_cols = app_params.col_names.length() > 0;

	// ======== first read the files ========
	stime = getSysTime();
	std::unordered_set<std::string> rows, columns;
	if (select_rows) {

		// read the file and put into a set.
		std::ifstream ifs(app_params.row_names);
		std::string line;
		// FMT_ROOT_PRINT("TF file: ");
		while (std::getline(ifs, line)) {
			rows.insert(splash::utils::trim(line));
			// FMT_ROOT_PRINT("{},", splash::utils::trim(line));
		}
		// FMT_ROOT_PRINT("\n   total {}\n", rows.size());
		ifs.close();
	}
	if (select_cols) {

		// read the file and put into a set.
		std::ifstream ifs(app_params.col_names);
		std::string line;
		// FMT_ROOT_PRINT("TF file: ");
		while (std::getline(ifs, line)) {
			columns.insert(splash::utils::trim(line));
			// FMT_ROOT_PRINT("{},", splash::utils::trim(line));
		}
		// FMT_ROOT_PRINT("\n   total {}\n", rows.size());
		ifs.close();
	}
	etime = getSysTime();
	FMT_ROOT_PRINT("Load TF data in {} sec\n", get_duration_s(stime, etime));
	
	// =========== check each gene/sample name, make a bool vector, and add matched to a list. 
	stime = getSysTime();
	std::string name;
	if (select_rows) {
		// now check for each gene whether it's in TF list.
		for (size_t i = 0; i < genes.size(); ++i) {
			name = splash::utils::trim(genes[i]);
			bool res = rows.count(name) > 0;
			// FMT_PRINT("\"{}\"\t{}\t{}\n", splash::utils::trim(genes[i]), genes[i].length(), (res ? "yes" : "no"));
			row_selected.push_back(res); // if this is one of the target TF, then safe it
			if (res) {
				r_names.push_back(name);
			}
		}
	}
	if (select_cols) {
		// now check for each gene whether it's in TF list.
		for (size_t i = 0; i < samples.size(); ++i) {
			name = splash::utils::trim(samples[i]);
			bool res = columns.count(name) > 0;
			// FMT_PRINT("\"{}\"\t{}\t{}\n", splash::utils::trim(genes[i]), genes[i].length(), (res ? "yes" : "no"));
			col_selected.push_back(res); // if this is one of the target TF, then safe it
			if (res) {
				c_names.push_back(name);
			}
		}
	}
	etime = getSysTime();
	FMT_ROOT_PRINT("match names in {} sec\n", get_duration_s(stime, etime));

	// ======== print selected data ===============
	if (select_rows) {
		if (mpi_params.rank == 0) {
			FMT_ROOT_PRINT("Selected rows {}:", r_names.size());
			for (auto name : r_names) {
				FMT_ROOT_PRINT("{},", name.c_str());
			}
			FMT_ROOT_PRINT("\n");
			FMT_ROOT_PRINT("MISSING rows:");
			for (auto name : rows) {
				if (gns.count(splash::utils::trim(name)) == 0)
					FMT_ROOT_PRINT("{},", name.c_str());
			}
			FMT_ROOT_PRINT("\n");
		}
		FMT_ROOT_PRINT("rows specified {}, found {}\n", rows.size(), r_names.size());

		if (r_names.size() == 0) {
			return;
		}
	}
	if (select_cols) {
		if (mpi_params.rank == 0) {
			FMT_ROOT_PRINT("Selected cols {}:", c_names.size());
			for (auto name : c_names) {
				FMT_ROOT_PRINT("{},", name.c_str());
			}
			FMT_ROOT_PRINT("\n");
			FMT_ROOT_PRINT("MISSING cols:");
			for (auto name : rows) {
				if (smpls.count(splash::utils::trim(name)) == 0)
					FMT_ROOT_PRINT("{},", name.c_str());
			}
			FMT_ROOT_PRINT("\n");
		}
		FMT_ROOT_PRINT("cols specified {}, found {}\n", columns.size(), c_names.size());

		if (c_names.size() == 0) {
			return;
		}
	}

	//======== Now create final array ===========
	stime = getSysTime();
	MatrixType output;
	if (select_rows) {
		if (select_cols) {
			output.resize(r_names.size(), c_names.size()); 
		} else {
			output.resize(r_names.size(), samples.size());
		}
	} else {
		if (select_cols) {
			output.resize(genes.size(), c_names.size());
		} else {
			output.resize(genes.size(), samples.size());
		}
	}

	if (select_rows) {
		
		for (size_t i = 0, j = 0; i < genes.size(); ++i) {
			if (row_selected[i]) {

				if (select_cols) {
					for (size_t k = 0, l = 0; k < samples.size(); ++k) {
						if (col_selected[k] ) {
							output(j, l) = input(i, k);
							++l;
						}
					}
				} else {
					memcpy(output.data(j), input.data(i), input.column_bytes());
				}
				++j;
			}
		}
	} else if (select_cols) {
		for (size_t i = 0; i < genes.size(); ++i) {
			for (size_t k = 0, l = 0; k < samples.size(); ++k) {
				if (col_selected[k] ) {
					output(i, l) = input(i, k);
					++l;
				}
			}
		}

	} else {
		output = input;
	}
	etime = getSysTime();
	FMT_ROOT_PRINT("Select in {} sec\n", get_duration_s(stime, etime));

	// =============== WRITE OUT ==============
	// NOTE: rank 0 writes out.
	stime = getSysTime();
	FMT_PRINT("output size: {} x {}\n", output.rows(), output.columns());
	if (mpi_params.rank == 0) {
		if (select_rows) {
			if (select_cols) {
				write_matrix(common_params.output, "array", r_names, c_names, output);
			} else {
				write_matrix(common_params.output, "array", r_names, samples, output);
			}
		} else {
			if (select_cols) {
				write_matrix(common_params.output, "array", genes, c_names, output);
			} else {
				write_matrix(common_params.output, "array", genes, samples, output);
			}
		}
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

	return 0;
}
