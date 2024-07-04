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

#include "splash/ds/aligned_matrix.hpp"
#include "splash/kernel/kernel_base.hpp"

#include <functional>  // reference_wrapper
#include <set>

#include <type_traits>

namespace mcp { namespace kernel {

// apply dpi filter.
// make this part like a correlation - input row_x and row_z.  also need  i_xz 
// also need isTF - x xor z is TF, but not y, then keep (dmap = false)
//    if i_xy >= i_xz, delete edge xz
//    if i_yz >= i_xz, delete edge xz.   i.e. yz has more mutual information than xz.
// i.e. keep (dmap = false) if (i-xy < i_xz) || (i_yz < i_xz) || 1 TF.
// the TF is extra.  do it separately?

// compute mask for removing edges based on data processing inequality. 
// specifically, processing cannot introduce information, so MI of xz should be larger than either xy or yz.

template <typename IT, bool EQUALITY_EXCLUSION = true, bool SYMMETRY = true>
class dpi_kernel : public splash::kernel::inner_product_pos<IT, bool, splash::kernel::DEGREE::VECTOR, splash::kernel::DEGREE::VECTOR> {
	protected:
		double tol_multiplier;
	public:
		dpi_kernel(double const & tolerance = 0.1) : tol_multiplier(1.0 / (1.0 - tolerance)) {}
		virtual ~dpi_kernel() {}

        void copy_parameters(dpi_kernel const & other) {
            tol_multiplier = other.tol_multiplier;
        }

		// NOTE: we should always delete when x == z.  All triangles with vertex y are degenerate, and we should always remove self edge.
		// NOTE: we can also skip checking for y==x and y==z.  these form self edges mi_xx and mi_zz, respectively, and are degenerate triangles.  skip (++y) and go on to other triangles.
		inline virtual bool operator()(size_t const & x, size_t const & z, IT const * row_x, IT const * row_z, size_t const & count) const {
			if (x == z) return true;

			IT i_xz = row_x[z] * tol_multiplier;

			size_t y = 0;
			if (EQUALITY_EXCLUSION)
				for (; (y < count) && ((y == x) || (y == z) || (row_x[y] < i_xz) || (row_z[y] < i_xz)); ++y) {}
			else
				for (; (y < count) && ((y == x) || (y == z) || (row_x[y] <= i_xz) || (row_z[y] <= i_xz)); ++y) {}

			return (y < count);  // if exit the loop early, then both MIs are larger than i_xz, so return true to delete MIs
		}
};


// y==count:  output is false, should stay false after TF
// y < count:  output was true.  If (x xor z) && !y, output needs to change to false 
template <typename IT, bool EQUALITY_EXCLUSION = true>
class dpi_tf_kernel : public splash::kernel::inner_product_pos<IT, bool, splash::kernel::DEGREE::VECTOR, splash::kernel::DEGREE::VECTOR> {
	protected:
		IT tol_multiplier;
		std::vector<IT> TFs;

	public:
		dpi_tf_kernel(std::vector<IT> const & _TFs = std::vector<IT>(), IT const & tolerance = 0.1) : tol_multiplier(1.0 / (1.0 - tolerance)), TFs(_TFs) {}
		virtual ~dpi_tf_kernel() {}

        void copy_parameters(dpi_tf_kernel const & other) {
            tol_multiplier = other.tol_multiplier;
			TFs = other.TFs;
        }

		// NOTE: we should always delete when x == z.  All triangles with vertex y are degenerate, and we should always remove self edge.
		// NOTE: we can also skip checking for y==x and y==z.  these form self edges mi_xx and mi_zz, respectively, and are degenerate triangles.  skip (++y) and go on to other triangles.
		inline virtual bool operator()(size_t const & x, size_t const & z, IT const * row_x, IT const * row_z, size_t const & count) const {
			if (x == z) return true;

			IT i_xz = row_x[z] * tol_multiplier;
			bool tf_xz = (TFs[x] > 0) ^ (TFs[z] > 0);
			size_t y = 0;

			if (EQUALITY_EXCLUSION)
				for (; (y < count) && 
					((y == x) || (y == z) || (row_x[y] < i_xz) || (row_z[y] < i_xz) || 
					((TFs[y] <= 0) && tf_xz) ); ++y) {}
			else
				for (; (y < count) && 
					((y == x) || (y == z) || (row_x[y] <= i_xz) || (row_z[y] <= i_xz) || 
					((TFs[y] <= 0) && tf_xz) ); ++y) {}
			
						
			return (y < count);  // if exit the loop early, then both MIs are larger than i_xz, so return true to delete MIs
		}
};



}}
