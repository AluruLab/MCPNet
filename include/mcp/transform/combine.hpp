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

#include "splash/ds/aligned_matrix.hpp"
#include "splash/kernel/kernel_base.hpp"

#include <functional>  // reference_wrapper

#include <type_traits>

namespace wave { namespace kernel {

// apply dpi filter.
// make this part like a correlation - input row_x and row_z.  also need  i_xz 
// also need isTF - x xor z is TF, but not y, then keep (dmap = false)
//    if i_xy >= i_xz, delete edge xz
//    if i_yz >= i_xz, delete edge xz.   i.e. yz has more mutual information than xz.
// i.e. keep (dmap = false) if (i-xy < i_xz) || (i_yz < i_xz) || 1 TF.
// the TF is extra.  do it separately?

// compute mask for removing edges based on data processing inequality. 
// specifically, processing cannot introduce information, so MI of xz should be larger than either xy or yz.


template <typename IT>
class max_kernel : public splash::kernel::binary_op<IT, IT, IT, splash::kernel::DEGREE::VECTOR> {

	public:

		inline virtual void operator()(IT const * in, IT const * aux1, size_t const & count, IT * out) const {
#if defined(__INTEL_COMPILER)
#pragma vector aligned
#endif
#if defined(USE_SIMD)
#pragma omp simd
#endif
			for (size_t y = 0; y < count; ++y) {
				out[y] = std::max(in[y], aux1[y]);
			}
		}
};

template <typename IT>
class min_kernel : public splash::kernel::binary_op<IT, IT, IT, splash::kernel::DEGREE::VECTOR> {

	public:

		inline virtual void operator()(IT const * in, IT const * aux1, size_t const & count, IT * out) const {
#if defined(__INTEL_COMPILER)
#pragma vector aligned
#endif
#if defined(USE_SIMD)
#pragma omp simd
#endif
			for (size_t y = 0; y < count; ++y) {
				out[y] = std::min(in[y], aux1[y]);
			}
		}
};

template <typename IT, typename OT = IT>
class weighted_mean_kernel : public splash::kernel::binary_op<IT, IT, OT, splash::kernel::DEGREE::VECTOR> {
	protected:
		double alpha;
		double beta;

	public:
		weighted_mean_kernel(double const & a = 0.5, double const & b = 0.5) : alpha(a), beta(b) {}
		virtual ~weighted_mean_kernel() {};

        void copy_parameters(weighted_mean_kernel const & other) {
			this->alpha = other.alpha;
			this->beta = other.beta;	
        }

		inline virtual void operator()(IT const * in, IT const * aux1, size_t const & count, OT * out) const {
#if defined(__INTEL_COMPILER)
#pragma vector aligned
#endif
#if defined(USE_SIMD)
#pragma omp simd
#endif
			for (size_t y = 0; y < count; ++y) {
				out[y] = alpha * in[y] + beta * aux1[y];
			}
		}
};


template <typename IT, typename OT = IT>
class madd_kernel : public splash::kernel::binary_op<IT, IT, OT, splash::kernel::DEGREE::VECTOR> {
	protected:
		double coeff;

	public:
		madd_kernel(double const & c = 1.0) : coeff(c) {}
		virtual ~madd_kernel() {};

        void copy_parameters(madd_kernel const & other) {
			this->coeff = other.coeff;
	    }

		inline virtual void operator()(IT const * in, IT const * aux1, size_t const & count, OT * out) const {
#if defined(__INTEL_COMPILER)
#pragma vector aligned
#endif
#if defined(USE_SIMD)
#pragma omp simd
#endif
			for (size_t y = 0; y < count; ++y) {
				out[y] = in[y] + coeff * aux1[y];
			}
		}
};

template <typename IT, typename OT = IT>
class add_kernel : public splash::kernel::binary_op<IT, IT, OT, splash::kernel::DEGREE::VECTOR> {
	public:

		inline virtual void operator()(IT const * in, IT const * aux1, size_t const & count, OT * out) const {
#if defined(__INTEL_COMPILER)
#pragma vector aligned
#endif
#if defined(USE_SIMD)
#pragma omp simd
#endif
			for (size_t y = 0; y < count; ++y) {
				out[y] = in[y] + aux1[y];
			}
		}
};

template <typename IT, typename OT = IT>
class sub_kernel : public splash::kernel::binary_op<IT, IT, OT, splash::kernel::DEGREE::VECTOR> {
	public:

		inline virtual void operator()(IT const * in, IT const * aux1, size_t const & count, OT * out) const {
#if defined(__INTEL_COMPILER)
#pragma vector aligned
#endif
#if defined(USE_SIMD)
#pragma omp simd
#endif
			for (size_t y = 0; y < count; ++y) {
				out[y] = in[y] - aux1[y];
			}
		}
};

template <typename IT, typename OT = IT>
class multiply_kernel : public splash::kernel::binary_op<IT, IT, OT, splash::kernel::DEGREE::VECTOR> {
	public:

		inline virtual void operator()(IT const * in, IT const * aux1, size_t const & count, OT * out) const {
#if defined(__INTEL_COMPILER)
#pragma vector aligned
#endif
#if defined(USE_SIMD)
#pragma omp simd
#endif
			for (size_t y = 0; y < count; ++y) {
				out[y] = in[y] * aux1[y];
			}
		}
};
template <typename IT, typename OT = IT>
class scale_kernel : public splash::kernel::transform<IT, OT, splash::kernel::DEGREE::VECTOR> {
	protected:
		double coeff;

	public:
		scale_kernel(double const & c = 1.0) : coeff(c) {}
		virtual ~scale_kernel() {};

        void copy_parameters(scale_kernel const & other) {
			this->coeff = other.coeff;
	    }

		inline virtual void operator()(IT const * in, size_t const & count, OT * out) const {
#if defined(__INTEL_COMPILER)
#pragma vector aligned
#endif
#if defined(USE_SIMD)
#pragma omp simd
#endif
			for (size_t y = 0; y < count; ++y) {
				out[y] = coeff * in[y];
			}
		}
};


// compute the ratio between 2 matrices.   used with dpi2_maxmin_kernel
template <typename IT, typename OT = IT>
class ratio_kernel : public splash::kernel::binary_op<IT, IT, OT, splash::kernel::DEGREE::VECTOR> {
	public:
		inline virtual void operator()(IT const * in, IT const * aux1, size_t const & count, OT * out) const {
			for (size_t y = 0; y < count; ++y) {
				out[y] = (std::abs(in[y])  < std::numeric_limits<IT>::epsilon()) ? 0.0 :  // explicitly prevent 0/0 = nan  case.
					(std::abs(aux1[y]) < std::numeric_limits<IT>::epsilon()) ? std::numeric_limits<OT>::max() :  // prevent x/0 = inf case. 
					// note:  aux1[y] can easily be zero for diagonal.
					static_cast<OT>(static_cast<double>(in[y]) / static_cast<double>(aux1[y]));
			}
		}
};



// apply one of the above using innerproduct pattern.   The output is a boolean matrix.
// then use threshold to apply boolean matrix to input matrix.


}}