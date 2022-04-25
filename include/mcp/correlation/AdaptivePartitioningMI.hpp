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
 * Author(s): Yongchao Liu, Tony C. Pan
 */

#pragma once

#include <vector>

#include "splash/kernel/kernel_base.hpp"
#include "splash/utils/memory.hpp"
#include "splash/ds/buffer.hpp"
#include "splash/kernel/random.hpp"

#include <math.h>
#include <type_traits>
#include <deque>


#if defined(USE_SIMD)
#include <omp.h>
#endif

namespace mcp { namespace kernel {


template <typename IT, typename Distribution, typename Generator>
class add_noise {
    protected:
        splash::kernel::random_number_generator<Generator> & generators;
        IT mn;
        IT mx;
		Distribution distribution;
        splash::utils::partitioner1D<PARTITION_EQUAL> partitioner;  // row partitioned.

	public:
		add_noise() {};
        add_noise(splash::kernel::random_number_generator<Generator> & _gen, IT const & min = 0.0, IT const & max = 1.0) : 
            generators(_gen), mn(min), mx(max), distribution(min, max) {}
        virtual ~add_noise() {}
        void copy_parameters(add_noise const & other) {
            generators = other.generators;
            mn = other.mn;
            mx = other.mx;
			distribution = other.distribution;
        }

        inline void operator()(splash::ds::aligned_matrix<IT> const & input, splash::ds::aligned_matrix<IT> & output) {
            splash::utils::partition<size_t> part(0, input.rows(), 0);
            this->operator()(part, input.columns(), input.column_bytes(), input.data(), output.data());
        }
        inline void operator()(splash::ds::aligned_matrix<IT> const & input, splash::ds::aligned_matrix<IT> & output,
            splash::utils::partition<size_t> const & part) {
            this->operator()(part, input.columns(), input.column_bytes(), input.data(), output.data());
        }

		inline virtual void operator()(size_t const & rows, size_t const & cols, size_t const & stride_bytes,
            IT const * input, IT * output) {
            splash::utils::partition<size_t> part(0, rows, 0);
            this->operator()(part, cols, stride_bytes, input, output);
        }

		inline void operator()(splash::utils::partition<size_t> const & part, size_t const & cols, size_t const & stride_bytes,
            IT const * input, IT * output) {
				FMT_PRINT("ADDING NOISE.\n");
				
			size_t thread_id = 0;
			size_t num_threads = 1;
#ifdef USE_OPENMP
#pragma omp parallel
            {
				thread_id = omp_get_thread_num();
				num_threads = omp_get_num_threads();
#endif
            // get the per-thread generator
            auto generator = generators.get_generator(thread_id);

            // get the partition
            splash::utils::partition<size_t> p = partitioner.get_partition(part, num_threads, thread_id);

            // compute. lambda capture is by reference.
            IT const * in;
			IT * out;
            size_t off = p.offset;
            for (size_t i = 0; i < p.size; ++i, ++off) {
                in = reinterpret_cast<IT const *>(reinterpret_cast<unsigned char const *>(input) + off * stride_bytes);
                out = reinterpret_cast<IT *>(reinterpret_cast<unsigned char *>(output) + off * stride_bytes);

				for(size_t i = 0; i < cols; ++i){
					out[i] = in[i] + distribution(generator);
				}
            }

            // compute. lambda capture is by reference.
#ifdef USE_OPENMP
#pragma omp barrier
			}
#endif
		};

};


// change for upper limit to be exclusive.
template <typename T, bool EXCLUSIVE = false>
struct Window
{
    static_assert(std::is_integral<T>::value, "support only integral input types");

	Window() : _x1(1), _y1(1), _x2(0), _y2(0), _npts(0), _whole(false) {}

	Window(T const & x1,T const & y1,T const & x2, T const & y2, T const & npts = 0, bool const & whole = false) :
        _x1(x1), _y1(y1), _x2(x2), _y2(y2), _npts(npts), _whole(whole) {}

    Window(Window const & other) = default;
    Window& operator=(Window const & other) = default;

	inline void set(T const & x1,T const & y1,T const & x2, T const & y2, T const & npts = 0, bool const & whole = false)
	{
		_x1 = x1;
        _y1 = y1;
        _x2 = x2;
        _y2 = y2;
		_npts = npts;
	}

	inline bool isDegenerate() // result of failed split.
	{
		return EXCLUSIVE ? ((_x1 >= _x2) || (_y1 >= _y2)) : ((_x1 > _x2) || (_y1 > _y2));
	}
	inline bool splittable()  // can be split
	{
		return (EXCLUSIVE ? (((_x1 + 1) < _x2) || ((_y1 + 1) < _y2)) : ((_x1 < _x2) || (_y1 < _y2))) && (_npts > 2);
	}
	inline bool computable()  // can compute for window.
	{
		return (_npts > 0);
	}


	void print()
	{
		FMT_PRINT_RT("{} {} {} {} {} {}\n", _x1, _y1, _x2, _y2, _npts, _whole ? "whole window" : "sub window");
	}

	T _x1;	/*lefttop x*/
	T _y1;	/*lefttop y*/
	T _x2;	/*rightbottom x*/  // exclusive
	T _y2;	/*rightbottom y*/  // exclusive
	T _npts;	/*number of points in this region*/
	bool _whole;
};



/*
 * TODO:
 *   [ ]  1. rowPtrX and Y offsets can be precomputed, once for the whole matrix.  colIndex can then be computed once per pairing.
 *   [ ]  2. lots of zero entries.  Not clear if this is right - all partitions returned 0?  log(1) = 0 or pxy = 0. 
 *             would computing marginal and join entropies be better?  (resolved, but not clear why. hdf5 write out issue?)
 *   [X]  3. use base2 log instead of natural log.
 */

template <typename IT, typename OT>
class AdaptivePartitionRankMIKernel: public splash::kernel::inner_product<IT, OT, splash::kernel::DEGREE::VECTOR> {
    // accept integer types as input (rank transformed)

    static_assert(std::is_integral<IT>::value, "support only integral input types");
    static_assert(std::is_floating_point<OT>::value, "support only floating point output types");
    
    protected:
        
		mutable splash::ds::buffer<IT> rowPtrX;
		mutable splash::ds::buffer<IT> rowPtrY;
		mutable splash::ds::buffer<IT> xGroupedByY;
		mutable splash::ds::buffer<Window<IT>> partitions;

		/*compute pairwise mutual information using adaptive partitioning estimator*/
		inline void _createSparseMatrix(IT const * vecX, IT const * vecY, size_t const & count) const
		{
			IT x, y, off;

            // NOTE: linear counting, lot of random accesses. would sort help?

			/*build a histogram and perform prefix sum for Y*/
            rowPtrY.resize(count + 1);
			for (size_t i = 0; i < count; ++i) {
				/*get the y coordinate*/
				y = vecY[i] + 1;   // shift index by 1, for prefix sum

				/*count the number of identical y coordinates*/
				++(rowPtrY.data[y]);
			}
			for (size_t i = 2; i <= count; ++i) {
                rowPtrY.data[i] += rowPtrY.data[i-1];
			}

			/*CSR sparse matrix*/
            rowPtrX.resize(count + 1);  // used as temp store here.
			// TODO: should xGroupedByY be cleared also?  prob don't need, but need to ensure resize anyways.
            xGroupedByY.resize(count + 1);
			for(size_t i = 0; i < count; ++i){
				/*get the y coordinate  == row*/
				y = vecY[i];  // not offset by 1 - used for indexing for Y offset.

				/*get the global offset of x*/
                // rowPtrY.data[y]: start of all (x, y) with same y.
                // rowPtrX.data[y]: offset within same y, of the last x encountered.
				off = rowPtrY.data[y] + rowPtrX.data[y];

				/*increase the local offset of x*/
				++(rowPtrX.data[y]);

				/*get and save the x coordinate*/
				xGroupedByY.data[off] = vecX[i];  // NOTE: this effectively stores vecX sorted by y.
			}

			/*build a histrogram and compute prefix sum for X*/
			rowPtrX.resize(count + 1);
			for(size_t i = 0; i < count; ++i){
				/*get the x coordinate*/
				x = vecX[i] + 1;

				/*count the number of x coordinate*/
				++(rowPtrX.data[x]);
			}
			for(size_t i = 2; i <= count; ++i){
				rowPtrX.data[i] += rowPtrX.data[i-1];
			}
		}

		inline size_t _getNumPointsInWindow(Window<IT>& window) const
		{
            /*check if the window is valid*/
			if(window.isDegenerate()){
				return 0;
			}

			size_t nx = rowPtrX.data[window._x2 + 1] - rowPtrX.data[window._x1];
			size_t ny = rowPtrY.data[window._y2 + 1] - rowPtrY.data[window._y1];
			if(nx == 0 || ny == 0){
				return 0;
			}

			/*check the sparse matrix*/
			size_t beg = rowPtrY.data[window._y1];
			size_t end = beg + ny;
			size_t n = 0;
            size_t x;
			for(size_t i = beg; i < end; i++){
				x = xGroupedByY.data[i];
				n += ((x >= window._x1) && (x <= window._x2));  // should not count on the border.  exclusive of the higher edge.
			}

			return n;
		}


    public:
		// AdaptivePartitionRankMIKernel(int const & _bins, int const & _numSamples) {}

        AdaptivePartitionRankMIKernel() {}

        void copy_parameters(const AdaptivePartitionRankMIKernel<IT,OT>& other){
        }

		virtual ~AdaptivePartitionRankMIKernel() {}

        inline OT operator()(IT const * first, IT const * second, size_t const & count) const  {

			size_t maxNumPartitions = count + 1;
            partitions.resize(maxNumPartitions);

			IT midX, midY, n;
			OT probX, probY, probXY;
			Window<IT> windows[4];
			const OT factor = 1.0L / static_cast<OT>(count);

			/*create sparse matrix*/
			_createSparseMatrix(first, second, count);

			/*adaptive partitioning estimator*/
			size_t popFrom = 0;
			size_t pushAt = 0;	/*empty partition list*/
			IT numValids, npts;

			/*Since there is no overlap between subwindows, we will have at most vecSize subwindows a total.
			Hence, a buffer of vecSize + 1 elements should be enough for the buffer*/
			/*push one partition from the back*/
			partitions.data[pushAt].set(0, 0, count - 1, count - 1, count);
			pushAt = (pushAt + 1) % maxNumPartitions;

			/*enter the core loop*/
			OT mi = 0;
			while(pushAt != popFrom){
				/*get the window*/
				Window<IT>& window = partitions.data[popFrom];
				//FMT_PRINT_RT("x1: {} y1: {} x2: {} y2: {} npts: {}\n", window._x1, window._y1, window._x2, window._y2, window._npts);

				/*split the window into 4*/
				midX = (window._x1 + window._x2) / 2;
				midY = (window._y1 + window._y2) / 2;

				/*boundary points are counted in for each subwindow*/
				windows[0].set(window._x1, window._y1, midX, midY);
				windows[1].set(midX + 1, window._y1, window._x2, midY);
				windows[2].set(window._x1, midY + 1, midX, window._y2);
				windows[3].set(midX + 1, midY + 1, window._x2, window._y2);

				/*get the number of points in each window*/
				n = 0;
				numValids = 0;
				for(int i = 0; i < 4; ++i){
					/*check how many points in the window*/
					npts = _getNumPointsInWindow(windows[i]);

					/*set the number of points within the window*/
					if(windows[i].isDegenerate()){
						windows[i]._npts = 0;
					}else{
						windows[i]._npts = npts;
						/*count the number of valid windows. This is used to avoid the cases
						that many samples have the same rank in their respective vector*/
						numValids++;
					}
					/*count the total number of points in the four sub-windows*/
					n += npts;
				}
				if(n != window._npts){
					FMT_PRINT_RT("ERROR: the window splitting has some bug ({} != {}): {} {} {} {}\n", n, window._npts, windows[0]._npts, windows[1]._npts, windows[2]._npts, windows[3]._npts);
					return 0.0;
				}

				/*compute chi-square test*/
				OT e = static_cast<OT>(n) / 4.0;
				OT chiSquare = 0, d;
				for(int i = 0; i < 4; ++i){
					d = static_cast<OT>(windows[i]._npts) - e;
					chiSquare += d * d;
				}
				chiSquare /= e;

				/*test the significance at P-value = 0.05 (freedom degree 3). If there is only sub-window is valid,
				it means that this window cannot be further split any more but contains > 4 points*/
				if(chiSquare < 7.815 || numValids == 1){
					/*already uniform within this window*/
                    probX = factor * static_cast<OT>(rowPtrX.data[window._x2 + 1] - rowPtrX.data[window._x1]);
					probY = factor * static_cast<OT>(rowPtrY.data[window._y2 + 1] - rowPtrY.data[window._y1]);
					probXY = factor * n;

					/*sum up this window*/
					mi += probXY * log2(probXY / (probX * probY));

					/*pop out the current*/
					popFrom = (popFrom + 1) % maxNumPartitions;
				}else{

					/*pop out the current*/
					popFrom = (popFrom + 1) % maxNumPartitions;

					/*if the number of points in a window is < 4, already uniform as per chi-square test no matter how they are distributed*/
					for(int i = 0; i < 4; ++i){
						/*empty windows and windows of size 1 (i.e. single overlapped points) are excluded*/
						if(windows[i]._npts == 0){
							continue;
						}
						if(windows[i]._npts < 4){
							/*compute the mutual information.*/
							probX = factor * static_cast<OT>(rowPtrX.data[windows[i]._x2 + 1] - rowPtrX.data[windows[i]._x1]);
							probY = factor * static_cast<OT>(rowPtrY.data[windows[i]._y2 + 1] - rowPtrY.data[windows[i]._y1]);
							probXY = factor * windows[i]._npts;

							/*sum up this window*/
							mi += probXY * log2(probXY / (probX * probY));
						}else{
							/*push one partition from back*/
							partitions.data[pushAt] = windows[i];
							pushAt = (pushAt + 1) % maxNumPartitions;

							/*check if the buffer is full*/
							if(pushAt == popFrom){
								FMT_PRINT_RT("ERROR: the window queue is impossible to be full\n");
                                return 0.0;
							}
						}
					}
				}
			}

			return mi;
        }

};


/*
 * TODO:
 *   [DONE]  1. cHistRowPtrX and Y offsets can be precomputed, once for the whole matrix.  colIndex can then be computed once per pairing.
 */

template <typename IT, long firstRank = 0>
class IntegralCumulativeHistogramKernel : public splash::kernel::transform<IT, IT, splash::kernel::DEGREE::VECTOR> {
    public:
        static_assert(std::is_integral<IT>::value, "support only integral input types");

		virtual ~IntegralCumulativeHistogramKernel() {}

		// output should be count+1 in size.
        inline void operator()(IT const * __restrict__ in_vec, 
            size_t const & count,
            IT * __restrict__ out_vec) const {
            /*build a histogram and perform prefix sum for Y*/

            IT y;
			for (size_t i = 0; i < count; ++i) {
				/*get the y coordinate*/
				y = in_vec[i] - firstRank + 1;   // first rank value is 0, so shift index by 1, for prefix sum

				/*count the number of identical y coordinates*/
				++(out_vec[y]);
			}
			for (size_t i = 2; i <= count; ++i) {
                out_vec[i] += out_vec[i-1];
			}

        }


};

// this version uses a precomputed cHistRowPtrX and cHistRowPtrY (cumulative histogram)
template <typename IT, typename OT, long firstRank = 0>
class AdaptivePartitionRankMIKernel2 : public splash::kernel::inner_product_pos<IT, OT, splash::kernel::DEGREE::VECTOR> {
    // accept integer types as input (rank transformed)

    static_assert(std::is_integral<IT>::value, "support only integral input types");
    static_assert(std::is_floating_point<OT>::value, "support only floating point output types");
    
    protected:
    
        IT const * cHistRowPtrs;
        size_t cHistRowPtrs_stride_bytes;

        mutable splash::ds::buffer<IT> temp;
		mutable splash::ds::buffer<IT> xGroupedByY;
		mutable std::deque<Window<IT, true>> partitions;  // windows have exclusive upper bound.

		/*compute pairwise mutual information using adaptive partitioning estimator*/
		inline void _createSparseMatrix(IT const & xx, IT const & yy, IT const * xRanks, IT const * yRanks, size_t const & count) const
		{
			IT y, off;

            // NOTE: linear counting, lot of random accesses. would sort help?
            IT const * cHistRowPtrY = reinterpret_cast<IT const *>(reinterpret_cast<unsigned char const *>(cHistRowPtrs) + yy * cHistRowPtrs_stride_bytes);

			/*CSR sparse matrix*/
            temp.resize(count + 1);  // used as temp store here.
			// TODO: should xGroupedByY be cleared also?  prob don't need, but need to ensure resize anyways.
            xGroupedByY.resize(count + 1);

			for (size_t i = 0; i < count; ++i){
				/*get the y coordinate  == row*/
				y = yRanks[i] - firstRank;  // Ranking is done with first rank of 0.

				/*get the global offset of x*/
                // cHistRowPtrY[y]: start of all (x, y) with same y.
                // temp.data[y]: offset within same y, of the last x encountered.
				off = cHistRowPtrY[y] + temp.data[y];


				/*increase the local offset of x*/
				temp.data[y] += 1;

				/*get and save the x coordinate*/
				xGroupedByY.data[off] = xRanks[i];  // NOTE: this effectively stores xRanks sorted by corresponding y rank..

				// if ((xx == 0) && (yy == 3))
				//  	FMT_PRINT("i, x, y, hist, temp, off, xGroupByY: {},{},{},{},{},{},{}\n", i, xRanks[i], yRanks[i], cHistRowPtrY[y], temp.data[y]-1, off, xGroupedByY.data[off]);
			}
		}

		inline size_t _getNumPointsInWindow(IT const & xx, IT const & yy, Window<IT, true>& window) const
		{
            /*check if the window is valid*/
			if(window.isDegenerate()){
				return 0;
			}
            IT const * cHistRowPtrX = reinterpret_cast<IT const *>(reinterpret_cast<unsigned char const *>(cHistRowPtrs) + xx * cHistRowPtrs_stride_bytes);
            IT const * cHistRowPtrY = reinterpret_cast<IT const *>(reinterpret_cast<unsigned char const *>(cHistRowPtrs) + yy * cHistRowPtrs_stride_bytes);

			IT nx = cHistRowPtrX[window._x2] - cHistRowPtrX[window._x1];
			IT ny = cHistRowPtrY[window._y2] - cHistRowPtrY[window._y1];
			if(nx == 0 || ny == 0) {  // if either 0 means no nxy.
				return 0;
			}

			/*check the sparse matrix*/
			IT beg = cHistRowPtrY[window._y1];
			IT end = beg + ny;
			size_t n = 0;
            IT x;
			for(IT i = beg; i < end; i++){
				x = xGroupedByY.data[i];
				n += ((x >= window._x1) && (x < window._x2));  // should not count on the border.  exclusive of the higher edge.
			}
			// if ((xx == 0) && (yy == 3) && (window._x1 <= 1160) && (window._y1 <= 970) && (window._x2 > 1160) && (window._y2 > 970) )
			// 	FMT_PRINT("beg end, x1, x2, y1, y2, nx, ny, n: {},{},{},{},{},{},{},{},{}\n", beg, end, window._x1, window._x2, window._y1, window._y2, nx, ny, n);

			return n;
		}

		double compute_for_window(IT const * cHistRowPtrX, IT const * cHistRowPtrY, Window<IT, true> const & win) const {
			/*already uniform within this window*/
			IT nx = cHistRowPtrX[win._x2] - cHistRowPtrX[win._x1];
			IT ny = cHistRowPtrY[win._y2] - cHistRowPtrY[win._y1];
			// probX = factor * static_cast<OT>(nx);
			// probY = factor * static_cast<OT>(ny);
			IT nxy = win._npts;

			/*sum up this window*/
			// mi += probXY * log2(probXY / (probX * probY));
			return static_cast<double>(nxy) * log2(static_cast<double>(nxy) / static_cast<double>(nx * ny));  
				//  px = nx/count, py = ny/count, pxy = nxy/count, so multiply by count.

		}

    public:
		AdaptivePartitionRankMIKernel2(IT const * _rowPtrs, size_t const & _rowPtrs_stride_bytes) :
            cHistRowPtrs(_rowPtrs), cHistRowPtrs_stride_bytes(_rowPtrs_stride_bytes) {}

        AdaptivePartitionRankMIKernel2() {}

        void copy_parameters(const AdaptivePartitionRankMIKernel2<IT,OT>& other){
            cHistRowPtrs = other.cHistRowPtrs;
            cHistRowPtrs_stride_bytes = other.cHistRowPtrs_stride_bytes;
        }

		virtual ~AdaptivePartitionRankMIKernel2() {}

        inline OT operator()(size_t const & xx, size_t const & yy, IT const * first, IT const * second, size_t const & count) const  {

            IT const * cHistRowPtrX = reinterpret_cast<IT const *>(reinterpret_cast<unsigned char const *>(cHistRowPtrs) + xx * cHistRowPtrs_stride_bytes);
            IT const * cHistRowPtrY = reinterpret_cast<IT const *>(reinterpret_cast<unsigned char const *>(cHistRowPtrs) + yy * cHistRowPtrs_stride_bytes);


			IT midX, midY;
			// OT probX, probY;
			// OT probXY;
			Window<IT, true> windows[4];

			/*create sparse matrix*/
			_createSparseMatrix(xx, yy, first, second, count);

			/*adaptive partitioning estimator*/
			IT npts;

			/*Since there is no overlap between subwindows, we will have at most vecSize subwindows a total.
			Hence, a buffer of vecSize + 1 elements should be enough for the buffer*/
			/*push one partition from the back*/
			partitions.emplace_back(0, 0, count, count, count, true);

			/*enter the core loop*/
			double mi = 0.0;
			while(!partitions.empty()){   // supposed to be a circular buffer.  but each windows spawns 4 windows with 0 or 1 elements....
				/*get the window*/
				Window<IT, true>& window = partitions.back();
				// FMT_PRINT_RT("x1: {} y1: {} x2: {} y2: {} npts: {}, queue {}\n", window._x1, window._y1, window._x2, window._y2, window._npts, partitions.size());
				
				// first split and compute chisquare

				/*split the window into 4*/
				midX = (window._x1 + window._x2) >> 1;
				midY = (window._y1 + window._y2) >> 1;

				/*boundary points are counted in for each subwindow*/
				windows[0].set(window._x1, window._y1, midX, midY);
				windows[1].set(midX, window._y1, window._x2, midY);
				windows[2].set(window._x1, midY, midX, window._y2);
				windows[3].set(midX, midY, window._x2, window._y2);

				/*get the number of points in each window*/
				// n = 0;
				// numValids = 0;
				for(int i = 0; i < 4; ++i){
					/*check how many points in the window*/
					npts = _getNumPointsInWindow(xx, yy, windows[i]);
					windows[i]._npts = npts;
				}

				/*compute chi-square test*/
				// note if there are 4 points, and all are in 1 subwindow, then chiSqr = 12. ((4-1)^2 + 3 (1^2))/1 = 12
				//   3 pts all in 1 subwin:  ((3-0.75)^2 + 3 * (0.75^2)) / 0.75 = 9
				//  2 pts:  ((2-0.5)^2 + 3 * 0.5^2)/0.5 = 6.   
				// so threshold is 2.
				double e = static_cast<double>(window._npts) / 4.0;
				double chiSquare = 0.0, d;
				for(int i = 0; i < 4; ++i){
					d = static_cast<double>(windows[i]._npts) - e;
					chiSquare += d * d;
				}
				chiSquare /= e;


				/*test the significance at P-value = 0.05 (freedom degree 3). If there is only sub-window is valid,
				it means that this window cannot be further split any more but contains > 4 points*/
				if ((chiSquare > 7.815) || window._whole) {
					// pop current window before inserting new ones
					partitions.pop_back();

					/*if the number of points in a window is < 4, already uniform as per chi-square test no matter how they are distributed*/
					for(int i = 0; i < 4; ++i) {
						if (windows[i].splittable()) {
							partitions.emplace_back(windows[i]);

						} else if (windows[i].computable()) {
							mi += compute_for_window(cHistRowPtrX, cHistRowPtrY, windows[i]);
						}
					}
				} else if (window.computable()) {
					// if ((xx == 0) && (yy == 3) && (window._x1 == 1000) && (window._y1 == 750))
					// 	FMT_PRINT("window 1125,875 chiSquare is too low. {} window n {} subwins {} {} {} {}\n", chiSquare, window._npts, windows[0]._npts, windows[1]._npts, windows[2]._npts, windows[3]._npts );
					mi += compute_for_window(cHistRowPtrX, cHistRowPtrY, window);
					// pop current window after compute.
					partitions.pop_back();
				}

			}
			// if ((xx == 0) && (yy < 10))
			// 	FMT_PRINT("row-col {} {} mi, count, result {} {} {} {}\n", xx, yy, mi, count, mi / static_cast<double>(count) + log2(static_cast<double>(count)), static_cast<OT>(mi / static_cast<double>(count) + log2(static_cast<double>(count))));

			return mi / static_cast<double>(count) + log2(static_cast<double>(count));
        }

};



}}