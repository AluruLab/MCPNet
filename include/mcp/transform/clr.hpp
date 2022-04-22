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

#include <cmath>  // sqrt

#include "splash/kernel/kernel_base.hpp"
#include "splash/utils/precise_float.hpp"
#include <tuple>

#if defined(USE_SIMD)
#include <omp.h>
#endif

namespace wave { namespace kernel { 

template <typename IT, typename IT2 = std::pair<IT, IT>, typename OT = IT>
class clr_kernel : public splash::kernel::ternary_op<IT, IT2, IT2, OT, splash::kernel::DEGREE::SCALAR> {
	
	public:
		using InputType = IT;
		using Input2Type = IT2;
        using OutputType = OT;

        inline virtual OT operator()(IT const & in,
			IT2 const & aux1,  IT2 const & aux2) const {
            
			OT zi = (std::abs(aux1.second) < std::numeric_limits<IT>::epsilon()) ?  0.0 : (in - aux1.first) / aux1.second;  
			OT zj = (std::abs(aux1.second) < std::numeric_limits<IT>::epsilon()) ?  0.0 : (in - aux2.first) / aux2.second;
			return std::sqrt(zi * zi + zj * zj);
        }
}; 


template <typename IT, typename IT2 = std::pair<IT, IT>, typename OT = IT>
class clr_vector_kernel : public splash::kernel::ternary_op<IT, IT2, IT2, OT, splash::kernel::DEGREE::VECTOR> {
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
				out_vector[j] = std::sqrt(zi * zi + zj * zj);  //stouffer's method
			}

        }
}; 


template <typename IT, typename IT2 = IT, typename OT = IT>
class zscored_clr_kernel : public splash::kernel::binary_op<IT, IT2, OT, splash::kernel::DEGREE::VECTOR> {
	
	public:
		using InputType = IT;
		using Input2Type = IT2;
        using OutputType = OT;

        inline virtual void operator()(IT const * zi,
			IT2 const * zj, size_t const & count, OT * out) const {
			for (size_t k = 0; k < count; ++k) {
				out[k] = std::sqrt(zi[k] * zi[k] + zj[k] * zj[k]);

			}
        }
}; 


}}