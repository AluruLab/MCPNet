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

#include <cmath>
#include "splash/ds/buffer.hpp"
#include "splash/utils/partition.hpp"
#include "splash/patterns/pattern.hpp"

namespace mcp { namespace stencil {

// compute pvalue via permutation.  Permutations is precomputed for the pattern instance.
// this version loops through the permute iterations at the outermost loop.
//   before permutation, the columns involved is preprocessed and saved.
//   in each iteration, all the rows (rows for each MPI rank) are permuted.
//  then the tiles are calculated via delegated call.
//   output of delegate is compared to target and count is incremented.
// final pv-value is generated after all permutation iterations.
//
// COMPARED TO DELEGATE PREPROCESSING:  this version does saves the column preprocessing once per iteration.
//
// requires copy of input1, intermediate storage of original inner product, permuted product, and a count.
//   for each row, shuffle occurs once per iteration regardless of number of tiles.
// more memory usage, but access should be more linear and less random (repeated shuffle). 
// assume output of Op is of OT type.
template <typename IT>
class Diagonal;

template<typename IT>
class Diagonal<splash::ds::aligned_matrix<IT> > {
    
    public:
        using InputType = splash::ds::aligned_matrix<IT>;
        using OutputType = splash::ds::aligned_matrix<IT>;

    protected:
        splash::utils::partitioner1D<PARTITION_EQUAL> partitioner;
	
        using part1D_type = splash::utils::partition<size_t>;

        int procs;
        int rank;

        IT target_value;
	
    public:
#ifdef USE_MPI
        Diagonal(IT const & target = 0, MPI_Comm comm = MPI_COMM_WORLD) :
            target_value(target) {
            MPI_Comm_size(comm, &procs);
            MPI_Comm_rank(comm, &rank);
        }
#else
        Diagonal(IT const & target = 0) :
            procs(1), rank(0), target_value(target) {
        }
#endif
        // input is full matrix.  output is distributed.
        void operator()(InputType const & input, OutputType & output) const {

            ///////// ------------- check sizes

            // --------- partition into tiles, the partition tiles for MPI
            auto stime = getSysTime();

            part1D_type mpi_rows = partitioner.get_partition(input.rows(), this->procs, this->rank);
            // FMT_PRINT_RT("MPI Rank {} partition: ", this->rank);
            // mpi_tile_parts.print("MPI TILEs: ");
            output.resize(mpi_rows.size, input.columns());


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
                // FMT_ROOT_PRINT_RT("partitioning info : {}, {}, {}\n", mpi_tile_parts.size, threads, thread_id);
                part1D_type omp_rows = partitioner.get_partition(mpi_rows, threads, thread_id);
                // FMT_PRINT_RT("thread {} partition: ", thread_id);
                // omp_rows.print("OMP PARTITION rows: ");

                // copy in the data
                // set up the preprocessed data storage and preprocess.
                size_t in_rid = omp_rows.offset;
                size_t out_rid = in_rid - mpi_rows.offset;
                for (size_t r = 0; r < omp_rows.size; ++r, ++in_rid, ++out_rid) {
                    memcpy(output.data(out_rid), input.data(in_rid), input.columns() * sizeof(IT));
                    output(out_rid, in_rid) = target_value;
                }
            }

            auto etime = getSysTime();
            FMT_ROOT_PRINT("Replaced Diag value in {} sec\n", get_duration_s(stime, etime));

        }

};

}}
