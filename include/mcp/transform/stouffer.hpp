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

#pragma once

// #include <cmath>  // sqrt

#include "splash/kernel/kernel_base.hpp"
#include "splash/utils/precise_float.hpp"
#include "splash/utils/math_utils.hpp" // sqrt

#if defined(USE_SIMD)
#include <omp.h>
#endif

namespace wave { namespace kernel { 


// if standard deviation is 0, then z score is zero (all values at mean)
template <typename IT, typename IT2 = std::pair<IT, IT>, typename OT = IT>
class stouffer_kernel : public splash::kernel::ternary_op<IT, IT2, IT2, OT, splash::kernel::DEGREE::SCALAR> {
	protected:
		static constexpr OT isqrt2 = 0.70710678118654752440084436210485L; // splash::utils::sqrt(0.5L);
		
	public:
		using InputType = IT;
		using Input2Type = IT2;
	        using OutputType = OT;

        inline virtual OT operator()(IT const & in,
			IT2 const & aux1,  IT2 const & aux2) const {

			// zscore
			OT zi = (std::abs(aux1.second) < std::numeric_limits<IT>::epsilon()) ?  0.0 : static_cast<OT>((in - aux1.first) / aux1.second);  
			OT zj = (std::abs(aux2.second) < std::numeric_limits<IT>::epsilon()) ?  0.0 : static_cast<OT>((in - aux2.first) / aux2.second);
			return (zi + zj) * isqrt2;  //stouffer's method
        }
}; 


template <typename IT, typename IT2 = std::pair<IT, IT>, typename OT = IT>
class stouffer_vector_kernel : public splash::kernel::ternary_op<IT, IT2, IT2, OT, splash::kernel::DEGREE::VECTOR> {
	protected:
		static constexpr OT isqrt2 = 0.70710678118654752440084436210485L; // splash::utils::sqrt(0.5L);
		
	public:
		using InputType = IT;
		using Input2Type = IT2;
        using OutputType = OT;

        inline virtual void operator()(IT const * in, IT2 const * aux1, IT2 const * aux2, 
			size_t const & count, OT * out_vector) const {

			IT2 aux = *aux1;
			OT denom = (std::abs(aux.second) < std::numeric_limits<IT>::epsilon()) ? 0.0 : (1.0 / aux.second);
			OT zi, zj;
			for (size_t j = 0; j < count; ++j) {
				// zscore
				zi = static_cast<OT>((in[j] - aux.first) * denom);  
				zj = (std::abs(aux2[j].second) < std::numeric_limits<IT>::epsilon()) ?  0.0 : static_cast<OT>((in[j] - aux2[j].first) / aux2[j].second);
				out_vector[j] = (zi + zj) * isqrt2;  //stouffer's method
			}

        }
}; 


template <typename IT, typename IT2 = IT, typename OT = IT>
class zscored_stouffer_kernel : public splash::kernel::binary_op<IT, IT2, OT, splash::kernel::DEGREE::VECTOR> {
	protected:
		static constexpr OT isqrt2 = 0.70710678118654752440084436210485L; // splash::utils::sqrt(0.5L);
		
	public:
		using InputType = IT;
		using Input2Type = IT2;
        using OutputType = OT;

        inline virtual void operator()(IT const * zi,
			IT2 const * zj, size_t const & count, OT * out) const {
			for (size_t k = 0; k < count; ++k) {
				out[k] = (zi[k] + zj[k]) * isqrt2;  //stouffer's method
			}

        }
}; 



template <typename IT, typename IT2 = std::pair<IT, IT>, typename OT = IT>
class weighted_stouffer_kernel : public splash::kernel::ternary_op<IT, IT2, IT2, OT, splash::kernel::DEGREE::SCALAR> {

	public:
		using InputType = IT;
		using Input2Type = IT2;
        using OutputType = OT;

        inline virtual OT operator()(IT const & in,
			IT2 const & aux1,  IT2 const & aux2) const {
			
			OT vari = aux1.second * aux1.second;
			OT varj = aux2.second * aux2.second;

			// numerator = sum of (zscore * var).   denome = sqrt of sum of square variances
			OT zi = (in - aux1.first) * aux1.second;  // zscore * var == zscore * stdev * stdev = (in - mean) * stdev
			OT zj = (in - aux2.first) * aux2.second;
			OT denom = sqrt(vari * vari + varj * varj);
			return (std::abs(denom) < std::numeric_limits<OT>::epsilon()) ? 0.0 : ((zi + zj) / denom);  // weighted stouffer's method
        }
}; 


}}
