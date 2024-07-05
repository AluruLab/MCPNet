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

template<typename DataType>
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
		bool mi_input;
        std::string coeffs;
		std::string groundtruth_list;
		std::string groundtruth_mat;
        std::string tf_input;
		bool clamped;
        bool run_evaluation;
		DataType diagonal;

		app_parameters() : mi_computes(mi_method_type::AP), groundtruth_list(""), 
                           groundtruth_mat(""), tf_input(""), clamped(false),
                           run_evaluation(false), diagonal(0.0)  {}
        app_parameters(const app_parameters<double>& other){
            computes.resize(other.computes.size());
            for(std::size_t i = 0; i < computes.size();++i){
               computes[i] = app_parameters<DataType>::method_type(other.computes[i]);
            }
            mi_computes = app_parameters<DataType>::mi_method_type(other.mi_computes);
            mi_input = other.mi_input;
            coeffs = other.coeffs;
            groundtruth_list = other.groundtruth_list;
            groundtruth_mat = other.groundtruth_mat;
            tf_input = other.tf_input;
            clamped = other.clamped;
            diagonal = DataType(other.diagonal);
        }
		virtual ~app_parameters() {}

		virtual void config(CLI::App& app) {
			auto mi_group = app.add_option_group("MI");
			mi_group->add_option(app.add_option("--mi-method", mi_computes, "MI Algorithm: 0 = Bspline, 1 (default) = Adaptive Partitioning"));
			mi_group->add_option(app.add_flag("--mi-input", mi_input, "Flag to indicate input file has precomputed MI"));
			mi_group->require_option(1);

			auto comp_opt = app.add_option("-m,--method", computes, "Algorithm: MCP2=1, MCP3=2, MCP4=3, EnsembleMCP=4")->group("MCP");
			auto clamped_opt = app.add_flag("--clamped", clamped, "output is clampped")->group("MCP");
			clamped_opt->needs(comp_opt);
			app.add_option("-f,--coefficients", coeffs, "file with combo coefficients.  For ensemble MCP only.")->group("MCP");

			auto gt_list_opt = app.add_option("-g,--groundtruth-list", groundtruth_list, "filename of groundtruth edge list.  require just forward or reverse to be present. (+) -> 1, (-) -> 0, unknown -> -1")->group("evaluation");
			auto gt_mat_opt = app.add_option("-x,--groundtruth-matrix", groundtruth_mat, "filename of groundtruth matrix. require symmetric matrix. (+) -> 1, (-) -> 0, unknown -> -1")->group("evaluation");
			app.add_option("--diagonal", diagonal, "Input MI matrix diagonal should be set as this value. default 0. if negative, use original MI")->group("MCP");

			app.add_option("--tf-input", tf_input, "transcription factor file, 1 per line.")->group("transcription factor");

			auto eval_opt = app.add_flag("--eval", run_evaluation, "Run evaluation")->group("MCP")->capture_default_str();
			auto opt_group = app.add_option_group("evaluation");
            opt_group->add_option(gt_list_opt);
            opt_group->add_option(gt_mat_opt);
            opt_group->require_option(1);
            opt_group->needs(eval_opt);
        

		}
		virtual void print(const char * prefix) const {
            int i = 1;
			FMT_ROOT_PRINT("{} MCP Method Size             : {}\n", prefix, computes.size());
			for (auto c : computes) {
				FMT_ROOT_PRINT("{} -> MCP Method {}            : {}\n", prefix, i, 
					(c == app_parameters<DataType>::method_type::MCP2 ? "MCP2" : 
					(c == app_parameters<DataType>::method_type::MCP3 ? "MCP3" : 
					(c == app_parameters<DataType>::method_type::MCP4 ? "MCP4" : 
					(c == app_parameters<DataType>::method_type::MU_MCP ? "Ensemble MCP" : 
					"unsupported"))))); 
                i++;
			}
			FMT_ROOT_PRINT("{} MI method                   : {}\n", prefix, 
				(mi_computes == BSpline ? "bspline" : 
				(mi_computes == AP ? "adaptive partitioning with ranking" : "unknown"))); 
            FMT_ROOT_PRINT("{} MI Input                    : {}\n", prefix, mi_input ? "Y" : "N"); 
            FMT_ROOT_PRINT("{} coefficient input           :  {}\n", prefix, coeffs.c_str()); 
			FMT_ROOT_PRINT("{} MCP compute clamping output : {}\n", prefix, (clamped ? "Y" : "N")); 
            FMT_ROOT_PRINT("{} groundtruth-list file       : {}\n", prefix, groundtruth_list.c_str()); 
            FMT_ROOT_PRINT("{} groundtruth-matrix file     : {}\n", prefix, groundtruth_mat.c_str()); 

            FMT_ROOT_PRINT("{} TF input                    : {}\n", prefix, tf_input.c_str()); 

			FMT_ROOT_PRINT("{} MI diagonal set to          : {}\n", prefix, diagonal); 
		}
        
};

template<typename DataType>
void run(splash::io::common_parameters& common_params,
         splash::io::mpi_parameters& mpi_params,
         app_parameters<DataType>& app_params ){


	// =============== SETUP INPUT ===================
	// NOTE: input data is replicated on all MPI procs.
	using MatrixType = splash::ds::aligned_matrix<DataType>;
	// using VectorType = splash::ds::aligned_vector<std::pair<DataType, DataType>>;
	MatrixType input;
	MatrixType mi;
	std::vector<std::string> genes;
	std::vector<std::string> samples;
	
	auto stime = getSysTime();
	auto etime = getSysTime();

	if (app_params.mi_input == false) {
		stime = getSysTime();
		// input = read_matrix<double>(common_params.input, "array", 
		// 		common_params.num_vectors, common_params.vector_size,
		// 		genes, samples, common_params.skip );
		input = read_matrix<DataType>(common_params.input, common_params.h5_group, 
			common_params.num_vectors, common_params.vector_size,
			genes, samples, 
            common_params.skip, 1, common_params.h5_gene_key,
            common_params.h5_samples_key, common_params.h5_matrix_key);
		etime = getSysTime();
		FMT_ROOT_PRINT("Loaded Data,EXP,,{},sec\n", get_duration_s(stime, etime));
	} else {
		stime = getSysTime();
		mi = read_matrix<DataType>(common_params.input, common_params.h5_group, 
				common_params.num_vectors, common_params.vector_size,
				genes, samples,
                common_params.skip, 1, common_params.h5_gene_key,
                common_params.h5_samples_key, common_params.h5_matrix_key);
		etime = getSysTime();
		FMT_ROOT_PRINT("Loaded MI,EXP,,{},sec\n", get_duration_s(stime, etime));
	}


	bool tfonly = app_params.tf_input.length() > 0;
	// load the tf gene names and max values
	std::vector<DataType> tfs;
	std::vector<std::string> tf_names;
	if (tfonly) {
		stime = getSysTime();
		load_tfs(app_params.tf_input, genes, tfs, tf_names, mpi_params.rank);
		if ((app_params.tf_input.length() > 0) && (tf_names.size() == 0)) {
			FMT_ROOT_PRINT("ERROR: no transcription factors in the specified file");
			return;
		}
		etime = getSysTime();
		FMT_ROOT_PRINT("Loaded,TF,,{},sec\n", get_duration_s(stime, etime));
	}

	stime = getSysTime();
	bool eval_by_list = app_params.run_evaluation && app_params.groundtruth_list.length() > 0;
	bool eval_by_mat = app_params.run_evaluation && app_params.groundtruth_mat.length() > 0;
	std::vector<std::tuple<std::string, std::string, int>> fulltruth, truth;
	splash::ds::aligned_matrix<char> fulltruth_mat, truth_mat;
	splash::ds::aligned_vector<DataType> pos;
	splash::ds::aligned_vector<DataType> neg;
	std::vector<std::pair<std::string, size_t>> dsrc_names;
	std::vector<std::pair<std::string, size_t>> dst_names;
	std::vector<std::tuple<size_t, size_t, int>> mask, mask2;
	mcp::kernel::aupr_kernel<DataType, char, DataType> auprkern;
	mcp::kernel::auroc_kernel<DataType, char, DataType> aurockern;

	size_t rows = common_params.num_vectors, columns = common_params.vector_size;
	std::vector<std::string> genes2;
	std::vector<std::string> samples2;

	if (tfonly) {
		if (eval_by_list) {
			fulltruth = read_groundtruth(app_params.groundtruth_list);
			// and filter
			filter_list_by_tfs(fulltruth, tf_names, truth);
		} else if (eval_by_mat) {
			fulltruth_mat = read_matrix<char>(app_params.groundtruth_mat, "array", 
				rows, columns,
				genes2, samples2);
			// and filter
			filter_mat_rows_by_tfs(fulltruth_mat, tfs, truth_mat);
		}
	} else {
		if (eval_by_list) {
			truth = read_groundtruth(app_params.groundtruth_list);
		} else if (eval_by_mat) {
			truth_mat = read_matrix<char>(app_params.groundtruth_mat, "array", 
				rows, columns,
				genes2, samples2);
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

	if (app_params.mi_input == false) {
	    if (app_params.mi_computes == app_parameters<DataType>::mi_method_type::BSpline) {
	    	output_prefix.append("_bs");
        } else {
	    	output_prefix.append("_ap");
        } 
    } else {
	    output_prefix.append("_inmi");
    }

	std::ostringstream oss;
	oss.str("");
	oss.clear();


	// =============== MI ===================
	MatrixType dmi;
	if (app_params.mi_input == false) {
		dmi.resize(input.rows(), input.columns());
		stime = getSysTime();
		if (app_params.mi_computes == app_parameters<DataType>::mi_method_type::BSpline)
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
		mask = select_from_list((tfonly ? dmi_tf.rows() : dmi.rows()),
			truth, (tfonly ? tf_names : genes), genes);
		etime = getSysTime();
		FMT_ROOT_PRINT("Computed,mi truth mask,,{},sec\n", get_duration_s(stime, etime));

	} else if (eval_by_mat) {
		stime = getSysTime();
		if (tfonly) {
			mask = select_tfs_from_mat(dmi_tf.rows(), tfs, truth_mat);
		} else
		mask = select_lower_triangle(dmi.rows(), truth_mat);
		etime = getSysTime();
		FMT_ROOT_PRINT("Computed,mi truth mask mat,,{},sec\n", get_duration_s(stime, etime));
	}

	if (eval_by_list || eval_by_mat) {
		stime = getSysTime();
		auto aupr = compute_aupr_auroc(mask, (tfonly ? dmi_tf : dmi), auprkern, aurockern, pos, neg);
		etime = getSysTime();
		FMT_ROOT_PRINT("Computed,auprroc {},,{},sec\n", aupr, get_duration_s(stime, etime));
	}

	if (app_params.mi_input == false) {
		stime = getSysTime();
		oss << output_prefix << ".mi.h5";
		write_matrix_distributed(oss.str(), "array", (tfonly ? tf_names : genes), genes, (tfonly ? dmi_tf : dmi));
		oss.str("");
		oss.clear();
		etime = getSysTime();
		FMT_ROOT_PRINT("Wrote,MI,,{},sec\n", get_duration_s(stime, etime));
		FMT_FLUSH();
	}

	// =============== MCP 2, 3, 4 and ensemble ========
	std::unordered_set<typename app_parameters<DataType>::method_type> comps;
	for (auto c : app_params.computes) {
		comps.insert(c);
	}



	MatrixType dratio;
	if (comps.find(app_parameters<DataType>::method_type::MCP2) != comps.end()) {
		// ------ ratio 1 --------
		stime = getSysTime();
		if (tfonly)
			compute_mcp2_tfs(mi_tf, mi, 1, tfs, dratio);
		else 
			compute_mcp2(mi, dratio);
		etime = getSysTime();
		FMT_ROOT_PRINT("Computed,MCP2,,{},sec\n", get_duration_s(stime, etime));


		// redo the masks based on rows of the output.
		if (eval_by_list) {

				stime = getSysTime();
			mask = select_from_list(dratio.rows(),
				truth, (tfonly ? tf_names : genes), genes);
				etime = getSysTime();
			FMT_ROOT_PRINT("Computed,mcp truth mask,,{},sec\n", get_duration_s(stime, etime));

		} else if (eval_by_mat) {
			stime = getSysTime();
			if (tfonly) {
				mask = select_tfs_from_mat(dratio.rows(), tfs, truth_mat);
			} else
				mask = select_lower_triangle(dratio.rows(), truth_mat);
			etime = getSysTime();
			FMT_ROOT_PRINT("Computed,mcp truth mask mat,,{},sec\n", get_duration_s(stime, etime));
		}


		if (eval_by_list || eval_by_mat) {
		stime = getSysTime();
			auto aupr = compute_aupr_auroc(mask, dratio, auprkern, aurockern, pos, neg);
			etime = getSysTime();
			FMT_ROOT_PRINT("Computed,auprroc {},,{},sec\n", aupr, get_duration_s(stime, etime));

		}


		stime = getSysTime();
		oss << output_prefix << ".mcp2.h5";
		write_matrix_distributed(oss.str(), "array", (tfonly ? tf_names : genes), genes, dratio);
		oss.str("");
		oss.clear();
		etime = getSysTime();
		FMT_ROOT_PRINT("Wrote,MCP2,,{},sec\n", get_duration_s(stime, etime));
		FMT_FLUSH();
	} 
	
	dratio.zero();
	if (comps.find(app_parameters<DataType>::method_type::MCP3) != comps.end()) {
		// ------ ratio 1 --------
		stime = getSysTime();
		if (tfonly)
			compute_mcp3_tfs(mi_tf, mi, 2, tfs, dratio);
		else 
			compute_mcp3(mi, dratio);
		etime = getSysTime();
		FMT_ROOT_PRINT("Computed,MCP3,,{},sec\n", get_duration_s(stime, etime));
		
		// redo the masks based on rows of the output.
		if (eval_by_list) {

			stime = getSysTime();
			mask = select_from_list(dratio.rows(),
				truth, (tfonly ? tf_names : genes), genes);
			etime = getSysTime();
			FMT_ROOT_PRINT("Computed,mcp truth mask,,{},sec\n", get_duration_s(stime, etime));

		} else if (eval_by_mat) {
			stime = getSysTime();
			if (tfonly) {
				mask = select_tfs_from_mat(dratio.rows(), tfs, truth_mat);
			} else
				mask = select_lower_triangle(dratio.rows(), truth_mat);
			etime = getSysTime();
			FMT_ROOT_PRINT("Computed,mcp truth mask mat,,{},sec\n", get_duration_s(stime, etime));
		}
		
		if (eval_by_list || eval_by_mat) {
			stime = getSysTime();
			auto aupr = compute_aupr_auroc(mask, dratio, auprkern, aurockern, pos, neg);
			etime = getSysTime();
			FMT_ROOT_PRINT("Computed,auprroc {},,{},sec\n", aupr, get_duration_s(stime, etime));
		}

		stime = getSysTime();
		oss << output_prefix << ".mcp3.h5";
		write_matrix_distributed(oss.str(), "array", (tfonly ? tf_names : genes), genes, dratio);
		oss.str("");
		oss.clear();
		etime = getSysTime();
		FMT_ROOT_PRINT("Wrote,MCP3,,{},sec\n", get_duration_s(stime, etime));
		FMT_FLUSH();
	} 

	dratio.zero();
	if (comps.find(app_parameters<DataType>::method_type::MCP4) != comps.end()) {
		// ------ ratio 1 --------
		stime = getSysTime();
		if (tfonly)
			compute_mcp4_tfs(mi_tf, mi, 3, tfs, dratio);
		else 
			compute_mcp4(mi, dratio);
		etime = getSysTime();
		FMT_ROOT_PRINT("Computed,MCP4,,{},sec\n", get_duration_s(stime, etime));

	// redo the masks based on rows of the output.
	if (eval_by_list) {

		stime = getSysTime();
		mask = select_from_list(dratio.rows(),
			truth, (tfonly ? tf_names : genes), genes);
		etime = getSysTime();
		FMT_ROOT_PRINT("Computed,mcp truth mask,,{},sec\n", get_duration_s(stime, etime));

	} else if (eval_by_mat) {
		stime = getSysTime();
		if (tfonly) {
			mask = select_tfs_from_mat(dratio.rows(), tfs, truth_mat);
		} else
			mask = select_lower_triangle(dratio.rows(), truth_mat);
		etime = getSysTime();
		FMT_ROOT_PRINT("Computed,mcp truth mask mat,,{},sec\n", get_duration_s(stime, etime));
	}

		if (eval_by_list || eval_by_mat) {
			stime = getSysTime();
			auto aupr = compute_aupr_auroc(mask, dratio, auprkern, aurockern, pos, neg);
			etime = getSysTime();
			FMT_ROOT_PRINT("Computed,auprroc {},,{},sec\n", aupr, get_duration_s(stime, etime));
		}

		stime = getSysTime();
		oss << output_prefix << ".mcp4.h5";
		write_matrix_distributed(oss.str(), "array", (tfonly ? tf_names : genes), genes, dratio);
		oss.str("");
		oss.clear();
		etime = getSysTime();
		FMT_ROOT_PRINT("Wrote,MCP4,,{},sec\n", get_duration_s(stime, etime));
		FMT_FLUSH();
	} 
	
	dratio.zero();
	// ============= compute combos ====================
	if (comps.find(app_parameters<DataType>::method_type::MU_MCP) != comps.end()) {
		// ----------- read the coefficient file.
		stime = getSysTime();
		MatrixType coeffs;
		size_t rs = 0;
		size_t cs = 0;
		{
			if (app_params.coeffs.length() > 0) {
				std::vector<std::string> rdummy;
				std::vector<std::string> cdummy;
				coeffs = read_matrix<DataType>(app_params.coeffs, "array", 
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


	// redo the masks based on rows of the output.
	if (eval_by_list) {

		stime = getSysTime();
		mask = select_from_list(dmaxmin1.rows(),
			truth, (tfonly ? tf_names : genes), genes);
		etime = getSysTime();
		FMT_ROOT_PRINT("Computed,mcp truth mask,,{},sec\n", get_duration_s(stime, etime));

	} else if (eval_by_mat) {
		stime = getSysTime();
		if (tfonly) {
			mask = select_tfs_from_mat(dmaxmin1.rows(), tfs, truth_mat);
		} else
			mask = select_lower_triangle(dmaxmin1.rows(), truth_mat);
		etime = getSysTime();
		FMT_ROOT_PRINT("Computed,mcp truth mask mat,,{},sec\n", get_duration_s(stime, etime));
	}

		// ----------- compute the combos
		auto stime_all = getSysTime();
		MatrixType dratiost;
		DataType* coeff = nullptr;
		oss << std::fixed << std::setprecision(3);

		// parameters for eval
		double max_aupr = 0, aupr;
		double max_saupr = 0, saupr;
		size_t co = 0, sco = 0;

		for (size_t i = 0; i < rs; ++i) {
			coeff = coeffs.data(i);

			if ((coeff[0] == 1) && (coeff[1] == 0) && (coeff[2] == 0) && (coeff[3] == 0)) continue;

			// ------ combo ratio --------
			stime = getSysTime();
			compute_combo((tfonly ? mi_tf : mi), dmaxmin1, dmaxmin2, dmaxmin3, coeff, dratio);
			etime = getSysTime();
			FMT_ROOT_PRINT("Computed,Combo,({}_{}_{}_{}),{},sec\n", 
				coeff[0], coeff[1], coeff[2], coeff[3], get_duration_s(stime, etime));

			if (eval_by_list || eval_by_mat) {
				stime = getSysTime();
				aupr = compute_aupr_auroc(mask, dratio, auprkern, aurockern, pos, neg);

				// save coeff if aupr is max.
				if (aupr > max_aupr) {
					max_aupr = aupr;
					co = i;
				}
				etime = getSysTime();
				FMT_ROOT_PRINT("Computed,Combo AUPR {},({}_{}_{}_{}),{},sec\n", 
					aupr, coeff[0], coeff[1], coeff[2], coeff[3], get_duration_s(stime, etime));
			} else {
				stime = getSysTime();
				oss << output_prefix << ".mcp_" << coeff[0] <<
					"_" << coeff[1] << "_" << coeff[2] << "_" << coeff[3] << ".h5";
				write_matrix_distributed(oss.str(), "array", (tfonly ? tf_names : genes), genes, dratio);
				oss.str("");
				oss.clear();
				etime = getSysTime();
				FMT_ROOT_PRINT("Wrote,Combo,({}_{}_{}_{}),{},sec\n", 
					coeff[0], coeff[1], coeff[2], coeff[3], get_duration_s(stime, etime));
			}

			// ------- stouffer --------
			stime = getSysTime();
			if (tfonly) {
				compute_stouffer_tf(dratio, dratiost); 
			} else {
			compute_stouffer(dratio, dratiost); 
			}
			etime = getSysTime();
			FMT_ROOT_PRINT("Computed,Combo Stouffer,({}_{}_{}_{}),{},sec\n", 
				coeff[0], coeff[1], coeff[2], coeff[3], get_duration_s(stime, etime));

			// write out again.
			if (eval_by_list || eval_by_mat) {
				
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
				FMT_ROOT_PRINT("Computed,Combo Stouffer AUPR {},({}_{}_{}_{}),{},sec\n", 
					saupr, coeff[0], coeff[1], coeff[2], coeff[3], get_duration_s(stime, etime));
			} else {
				stime = getSysTime();
				oss << output_prefix << ".mcpst_" << coeff[0] <<
					"_" << coeff[1] << "_" << coeff[2] << "_" << coeff[3] << ".h5";
				write_matrix_distributed(oss.str(), "array", (tfonly ? tf_names : genes), genes, dratiost);
				oss.str("");
				oss.clear();

				etime = getSysTime();
				FMT_ROOT_PRINT("Wrote,Combo Stouffer,({}_{}_{}_{}),{},sec\n", 
					coeff[0], coeff[1], coeff[2], coeff[3], get_duration_s(stime, etime));
			}
		}

		if (eval_by_list || eval_by_mat) {
			// ------ combo ratio --------
			stime = getSysTime();
			compute_combo((tfonly ? mi_tf : mi), dmaxmin1, dmaxmin2, dmaxmin3, coeffs.data(co), dratio);
			etime = getSysTime();
			FMT_ROOT_PRINT("Computed,MAX Combo aupr {},({}_{}_{}_{}),{},sec\n", 
				max_aupr, coeffs.at(co, 0), coeffs.at(co, 1), coeffs.at(co, 2), coeffs.at(co, 3), get_duration_s(stime, etime));
			stime = getSysTime();
			auto aupr = compute_aupr_auroc(mask, dratio, auprkern, aurockern, pos, neg);
			etime = getSysTime();
			FMT_ROOT_PRINT("Computed,MAX Combo aupr auroc {},,{},sec\n", aupr, get_duration_s(stime, etime));


			stime = getSysTime();
			// write out results with the max AUPR for regular and stouffer.
			oss << output_prefix << ".mcp_" << coeffs.at(co, 0) <<
				"_" << coeffs.at(co, 1) << "_" << coeffs.at(co, 2) << "_" << coeffs.at(co, 3) << ".h5";
			write_matrix_distributed(oss.str(), "array", (tfonly ? tf_names : genes), genes, dratio);
			oss.str("");
			oss.clear();
			etime = getSysTime();
			FMT_ROOT_PRINT("Wrote,MAX Combo,({}_{}_{}_{}),{},sec\n", 
				coeffs.at(co, 0), coeffs.at(co, 1), coeffs.at(co, 2), coeffs.at(co, 3), get_duration_s(stime, etime));


			// ------- stouffer --------
			stime = getSysTime();
			compute_combo((tfonly ? mi_tf : mi), dmaxmin1, dmaxmin2, dmaxmin3, coeffs.data(sco), dratio);
			if (tfonly) compute_stouffer_tf(dratio, dratiost); 
			else compute_stouffer(dratio, dratiost); 
			etime = getSysTime();
			FMT_ROOT_PRINT("Computed,MAX Combo AND Stouffer aupr {},({}_{}_{}_{}),{},sec\n", 
				max_saupr, coeffs.at(sco, 0), coeffs.at(sco, 1), coeffs.at(sco, 2), coeffs.at(sco, 3), get_duration_s(stime, etime));
			stime = getSysTime();
			auto saupr = compute_aupr_auroc(mask, dratiost, auprkern, aurockern, pos, neg);
			etime = getSysTime();
			FMT_ROOT_PRINT("Computed,MAX Combo AND Stouffer aupr auroc {},,{},sec\n", saupr, get_duration_s(stime, etime));

			stime = getSysTime();
			// write out results with the max AUPR for regular and stouffer.
			oss << output_prefix << ".mcpst_" << coeffs.at(sco, 0) <<
				"_" << coeffs.at(sco, 1) << "_" << coeffs.at(sco, 2) << "_" << coeffs.at(sco, 3) << ".h5";
			write_matrix_distributed(oss.str(), "array", (tfonly ? tf_names : genes), genes, dratiost);
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


}


int main(int argc, char* argv[]) {

	//==============  PARSE INPUT =====================
	CLI::App app{"MI Combo Net"};

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

    if(common_params.use_single) {
        app_parameters<float> flapp_params(app_params);
        run<float>(common_params, mpi_params, flapp_params);
    } else {
        run<double>(common_params, mpi_params, app_params);
    }
	return 0;
}
