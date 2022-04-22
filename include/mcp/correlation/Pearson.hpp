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

#include "splash/kernel/dotproduct.hpp"

#if defined(USE_SIMD)
#include <omp.h>
#endif

namespace wave { namespace correlation { 


template<typename IT, typename OT = IT>
class PearsonKernel : public splash::kernel::DotProductKernel<IT, OT> {

	public:
		inline OT operator()(IT const * first, IT const * second, size_t const & count) const  {
            return  splash::kernel::DotProductKernel<IT, OT>::operator()(first, second, count) / static_cast<IT>(count);
		};
};

}}