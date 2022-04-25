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
#include "mcp/filter/simd_maxmin.hpp"
#include "mcp/transform/combine.hpp"

#include <functional>  // reference_wrapper
#include <set>

#include <type_traits>

namespace mcp { namespace kernel {

// compute the max (along vector) of pairwise min between 2 vectors.  this will be used 2x.
template <typename IT, bool MASKED=false, bool ALLOW_LOOPS=false>
class mcp2_maxmin_kernel;

template <typename IT, bool MASKED>
class mcp2_maxmin_kernel<IT, MASKED, false> : public splash::kernel::inner_product_pos<IT, IT, splash::kernel::DEGREE::VECTOR, splash::kernel::DEGREE::VECTOR> {
	protected:
		std::vector<double> TFs;

	public:
		mcp2_maxmin_kernel(std::vector<double> const & _TFs = std::vector<double>()) : TFs(_TFs) {}
		virtual ~mcp2_maxmin_kernel() {}

        void copy_parameters(mcp2_maxmin_kernel const & other) {
        	TFs = other.TFs;
        }

		inline virtual IT operator()(size_t const & x, size_t const & z, IT const * row_x, IT const * row_z, size_t const & count) const {
			// original dpi code:  edge (x, z)  MI is mi_xz, with tolerance adjusted as i_xz = mi_xz / (1 - tol)
			// delete (x, z) if mi_xy >= i_xz AND mi_yz >= i_xz, for ANY y \in {0 .. count-1}.
			// i.e. xyz path transmits information better than xz. 
			// note all mi >= 0

			// dpi:  if all 3 are same and are max amongst all other interactions, all 3 are removed...
			
			// NOTE: we should always delete when x == z.  All triangles with vertex y are degenerate, and we should always remove self edge.
			// NOTE: we can also skip checking for y==x and y==z.  these form self edges mi_xx and mi_zz, respectively, and are degenerate triangles.  skip (++y) and go on to other triangles.
			// if (x == z) return static_cast<IT>(0.0);  // treat as if mi_xz == 0

			// note x and z entries are set to 0.  (self MI excluded).  Then the max is taken.
			double mx;
			if (MASKED) 
				mx = mcp::kernel::max_of_pairmin(row_x, row_z, count, x, z, TFs.data());
			else 
				mx = mcp::kernel::max_of_pairmin(row_x, row_z, count, x, z);
			return static_cast<IT>(mx);

		}
};
template <typename IT, bool MASKED>
class mcp2_maxmin_kernel<IT, MASKED, true> : public splash::kernel::inner_product<IT, IT, splash::kernel::DEGREE::VECTOR, splash::kernel::DEGREE::VECTOR> {
	protected:
		std::vector<double> TFs;

	public:
		mcp2_maxmin_kernel(std::vector<double> const & _TFs = std::vector<double>()) : TFs(_TFs) {}
		virtual ~mcp2_maxmin_kernel() {}

        void copy_parameters(mcp2_maxmin_kernel const & other) {
        	TFs = other.TFs;
        }

		inline virtual IT operator()(IT const * row_x, IT const * row_z, size_t const & count) const {
			double mx;
			if (MASKED) 
				mx = mcp::kernel::max_of_pairmin(row_x, row_z, count, TFs.data());
			else 
				mx = mcp::kernel::max_of_pairmin(row_x, row_z, count);
			return static_cast<IT>(mx);
		}
};


template <typename IT, typename OT, bool MASKED=false>
class mcp_ratio_kernel : public splash::kernel::inner_product_pos<IT, OT, splash::kernel::DEGREE::VECTOR, splash::kernel::DEGREE::VECTOR> {
	protected:
		mcp2_maxmin_kernel<IT, MASKED, false> maxmin;

	public:
		mcp_ratio_kernel(std::vector<double> const & _TFs = std::vector<double>()) : maxmin(_TFs) {}
		virtual ~mcp_ratio_kernel() {}

        void copy_parameters(mcp_ratio_kernel const & other) {
			maxmin.copy_parameters(other.maxmin);
        }


		inline virtual OT operator()(size_t const & x, size_t const & z, IT const * row_x, IT const * row_z, size_t const & count) const {
			if (std::abs(row_x[z]) < std::numeric_limits<IT>::epsilon()) return static_cast<OT>(0.0);  // treat as if mi_xz == 0
			double mx = maxmin(x, z, row_x, row_z, count);

			if (std::abs(mx) < std::numeric_limits<double>::epsilon()) return std::numeric_limits<OT>::max();  // all other edges have mi of 0 (but xz is not 0).  keep the edge always.
			
			return static_cast<OT>(static_cast<double>(row_x[z]) / mx);  // normal case.  if negative, that means always kept.  else should be between 0 and 1.

		}
};

/***
 * Computes the tolerance upper bound that deletes an edge via data processing inequality. 
 * To use the result, threshold with a user specified tolerance.  all entries greater than the user specified tolerances are kept.
 * NOTE: for ROC/PR curves, threshold is moved from smallest to largest.
 */
template <typename IT, typename OT, bool MASKED=false>
class mcp_tolerance_kernel : public splash::kernel::inner_product_pos<IT, OT, splash::kernel::DEGREE::VECTOR, splash::kernel::DEGREE::VECTOR> {
	protected:
		bool clamped;
		mcp_ratio_kernel<IT, OT, MASKED> ratio;

	public:
		mcp_tolerance_kernel(bool const & _clamped = false, std::vector<double> const & _TFs = std::vector<double>() ) : 
			clamped(_clamped || std::is_unsigned<OT>::value), ratio(_TFs) {}
		virtual ~mcp_tolerance_kernel() {}

        void copy_parameters(mcp_tolerance_kernel const & other) {
			ratio.copy_parameters(other.ratio);
			this->clamped = other.clamped;
        }


		inline virtual OT operator()(size_t const & x, size_t const & z, IT const * row_x, IT const * row_z, size_t const & count) const {
			// original dpi code:  edge (x, z)  MI is mi_xz, with tolerance adjusted as i_xz = mi_xz / (1 - tol)
			// delete (x, z) if mi_xy >= i_xz AND mi_yz >= i_xz, for ANY y \in {0 .. count-1}.
			// i.e. xyz path transmits information better than xz. 
			// note all mi >= 0

			// dpi:  if all 3 are same and are max amongst all other interactions, all 3 are removed...
			
			/**  to compute tolerance for a y that causes (x, z) to be deleted
			 * 		min(mi_xy, mi_yz) >= mi_xz / (1 - tol_xyz) 
			 *		so (1 - tol_xyz) >= mi_xz / min(mi_xy, mi_yz)
			 *		and tol_xyz <= (1 - mi_xz / min(mi_xy, mi_yz))
			 *   	i.e. any tolerance value tol_xyz at most (1 - mi_xz / min(mi_xy, mi_yz)) results in edge (x,z) deleted due to vertex y.
			 *   for (x, z) to be kept, the condition tol_xyz > (1 - mi_xz / min(mi_xy, mi_yz)) must hold for all y.
			 * 		i.e. tol_xz > max_y (1 - mi_xz / min(mi_xy, mi_yz)) for all y.
			 * 			tol_xz > 1 - min_y (mi_xz / min(mi_xy, mi_yz)) for all y
			 * 			tol_xz > 1 - mi_xz / max_y(min(mi_xy, mi_yz)) for all y
			 * 		any tolerance value tol <= tol_xz means there exists a y such that tol <= (1 - mi_xz / min(mi_xy, mi_yz)) to cause (x, z) to be deleted.
			 * 	
			 * 	IMPORTANT:  this kernel computes the minimum tolerance at or below which the edge is deleted
			 * 
			 * 	 tol_xz is therefore the minimum tolerance value above which the edge will remain after DPI
			 * 	   since MI values are non-negative, mi_xz / max_y(min(mi_xy, mi_yz)) has range [0.0, inf] (so max tol has range [-inf, 1].  user tol should be in [0, 1])
			 * 	   max_y(min(mi_xy, mi_yz)) == mi_xz > 0.0 -> tol_xz = 0.0.  edge is kept if tol > 0
			 * 	   mi_xz == 0 -> tol_xz = 1:  edge is deleted regardless of tolerance.
			 * 	   max_y(min(mi_xy, mi_yz)) == 0.0 -> tol_xz ~= -inf.  edge is always kept
			 *	   max_y(min(mi_xy, mi_yz)) > mi_xz > 0.0 -> tol_xz in range (0.0, 1.0), normal
			 * 	   mi_xz > max_y(min(mi_xy, mi_yz)) > 0.0 -> tol_xz < 0.0.  edge is always kept
			 * 	
			 *   a user specified tolerance tol, 
			 * 		tol > tol_xz -> keep (x, z)
			 * 		tol <= tol_xz -> there exists at least 1 y, tol <= tol_xz == tol_xyz == (1 - mi_xz / min(mi_xy, mi_yz)). so the (x, z) would be deleted.
			 *   
			 * tolerance range [0, 1].  0 is strict, so few marked as positive.  increasing tolerance -> less strict, so more marked as positive (true or false positive)
			 */      
			// NOTE: we should always delete when x == z.  All triangles with vertex y are degenerate, and we should always remove self edge.
			// NOTE: we can also skip checking for y==x and y==z.  these form self edges mi_xx and mi_zz, respectively, and are degenerate triangles.  skip (++y) and go on to other triangles.

			OT out = ratio(x, z, row_x, row_z, count);  // normal case.  if negative, that means always kept.  else should be between 0 and 1.
			out = this->clamped ? std::min(static_cast<OT>(1.0), out) : out;
			return static_cast<OT>(1.0) - out;

		}
};



/***
 * Computes a ratio based on the idea of DPI with some tolerance:
 * 		mi_xz - max_y(min(mi_xy, mi_yz)).
 * range of value is [-max .. max], with 0 indicating equality.  
 * 	negative values imply DPI is not satisfied.
 *  positive values imply DPI is satisfied
 *  at 0, exact equality is achieved.
 *  thresholding at other values implies a shift (tolerance)
 * 
 */
template <typename IT, typename OT, bool MASKED=false>
class mcp_diff_kernel : public splash::kernel::inner_product_pos<IT, OT, splash::kernel::DEGREE::VECTOR, splash::kernel::DEGREE::VECTOR> {
	protected:
		bool clamped;
		mcp2_maxmin_kernel<IT, MASKED, false> maxmin;

	public:
		mcp_diff_kernel(std::vector<double> const & _TFs = std::vector<double>()) : clamped(std::is_unsigned<OT>::value), maxmin(_TFs) {}
		virtual ~mcp_diff_kernel() {};

        void copy_parameters(mcp_diff_kernel const & other) {
			this->clamped = other.clamped;
			maxmin.copy_parameters(other.maxmin);
        }

		inline virtual OT operator()(size_t const & x, size_t const & z, IT const * row_x, IT const * row_z, size_t const & count) const {
			double mx = maxmin(x, z, row_x, row_z, count);			
			double out = static_cast<double>(row_x[z]) - mx;
			return static_cast<OT>(clamped ? std::max(0.0, out) : out); 
			
		}
};



// compute the max (along vector) of pairwise min between 2 vectors.  this will be used 2x.
template <typename IT, bool ALLOW_LOOP=false>
class maxmin3_kernel;

template <typename IT>
class maxmin3_kernel<IT, false> : public splash::kernel::inner_product_pos<IT, IT, splash::kernel::DEGREE::VECTOR, splash::kernel::DEGREE::VECTOR> {
	protected:
		::splash::ds::aligned_matrix<IT> const * matrix;
		int transition;
		std::vector<double> const * TFs;

	public:
		maxmin3_kernel() {};
		maxmin3_kernel(splash::ds::aligned_matrix<IT> const * mat, int const & _transition = -1, std::vector<double> const * _TFs = nullptr) : matrix(mat), transition(_transition), TFs(_TFs) {}
		virtual ~maxmin3_kernel() {}

        void copy_parameters(maxmin3_kernel const & other) {
			matrix = other.matrix;
			transition = other.transition;
        	TFs = other.TFs;
        }

		inline virtual IT operator()(size_t const & x, size_t const & z, IT const * row_x, IT const * row_z, size_t const & count) const {
			// note x and z entries are set to 0.  (self MI excluded).  Then the max is taken.
			if (matrix == nullptr) return 0;
			if (x == z) return 0;

			// naive approach.  use a double loop for x-i-j-z, iterating over all r and s.
			// note, skip x, and z.
			double mx = std::numeric_limits<double>::lowest(); // for max over all i, j, of min(xi, ij, jz)
			double mn;  //  min(xi, ij, jz)
			if ((transition > 0) && (TFs != nullptr) && (TFs->size() >= count)) {
				// transition indicate where in the path we switch to all genes.  in otherwords, when the TFs membership check stops.
				// skip row_x[x], row_x[z], row_z[x], row_z[z]
				for (size_t i = 0; i < count; ++i) {
					// get the minimum along path x-i-j-z
					if ((i == x) || (i == z)) continue;
					if ((transition >= 1) && ((*TFs)[i] < 0)) continue;  // skip if not TF and ttgg tttg
					for (size_t j = 0; j < count; ++j) {
						if ((j == x) || (j == z)) continue;
						if (i == j) continue;
						if ((transition >= 2) && ((*TFs)[j] < 0)) continue;  // skip if not TF and tttg

						// compute the min of xijz
						mn = std::min(row_x[i], (*matrix)(i, j));
						mn = std::min(mn, row_z[j]);
						// accumulate the max
						mx = std::max(mx, mn);
					}
				}

			} else { // 0 or less:  t-g-g-g or g-g-g-g
				// skip row_x[x], row_x[z], row_z[x], row_z[z]
				IT mmx;
				std::set<size_t> excl;
				excl.insert(x);
				excl.insert(z);
				for (size_t i = 0; i < count; ++i) {
					// get the minimum along path x-i-j-z
					if ((i == x) || (i == z)) continue;
					excl.insert(i);
					// // max_j(min(xi, ij, jz))) = min(xi, max_j(min(ij, jz)))
					mmx = mcp::kernel::max_of_pairmin(matrix->data(i), row_z, count, excl);   // TODO: use a mask to mask out multiple entries.
					mn = std::min(row_x[i], mmx);
					mx = std::max(mx, mn);
					excl.erase(i);

					// mati = matrix->data(i);
					// for (size_t j = 0; j < count; ++j) {
					// 	if ((j == x) || (j == z)) continue;
					// 	if (i == j) continue;

					// 	// compute the min of xijz
					// 	mn = std::min(xi, row_z[j]);
					// 	mn = std::min(mn, mati[j]);
					// 	// accumulate the max
					// 	mx = std::max(mx, mn);
					// }
				}
			}
			return static_cast<IT>(mx);
		}
};
template <typename IT>
class maxmin3_kernel<IT, true> : public splash::kernel::inner_product<IT, IT, splash::kernel::DEGREE::VECTOR, splash::kernel::DEGREE::VECTOR> {
	protected:
		::splash::ds::aligned_matrix<IT> const * matrix;
		int transition;
		std::vector<double> const * TFs;

	public:
		maxmin3_kernel() {};
		maxmin3_kernel(splash::ds::aligned_matrix<IT> const * mat, int const & _transition = -1, std::vector<double> const * _TFs = nullptr) : matrix(mat), transition(_transition), TFs(_TFs) {}
		virtual ~maxmin3_kernel() {}

        void copy_parameters(maxmin3_kernel const & other) {
			matrix = other.matrix;
			transition = other.transition;
        	TFs = other.TFs;
        }

		inline virtual IT operator()(IT const * row_x, IT const * row_z, size_t const & count) const {
			// note x and z entries are set to 0.  (self MI excluded).  Then the max is taken.
			if (matrix == nullptr) return 0;

			// naive approach.  use a double loop for x-i-j-z, iterating over all r and s.
			// note, skip x, and z.
			double mx = std::numeric_limits<double>::lowest(); // for max over all i, j, of min(xi, ij, jz)
			double mn;  //  min(xi, ij, jz)
			if ((transition > 0) && (TFs != nullptr) && (TFs->size() >= count)) {
				// transition indicate where in the path we switch to all genes.  in otherwords, when the TFs membership check stops.
				// skip row_x[x], row_x[z], row_z[x], row_z[z]
				for (size_t i = 0; i < count; ++i) {
					// get the minimum along path x-i-j-z
					if ((transition >= 1) && ((*TFs)[i] < 0)) continue;  // skip if not TF and ttgg tttg
					for (size_t j = 0; j < count; ++j) {
						if ((transition >= 2) && ((*TFs)[j] < 0)) continue;  // skip if not TF and tttg

						// compute the min of xijz
						mn = std::min(row_x[i], (*matrix)(i, j));
						mn = std::min(mn, row_z[j]);
						// accumulate the max
						mx = std::max(mx, mn);
					}
				}

			} else { // 0 or less:  t-g-g-g or g-g-g-g
				// skip row_x[x], row_x[z], row_z[x], row_z[z]
				IT mmx;
				for (size_t i = 0; i < count; ++i) {
					// get the minimum along path x-i-j-z
					// // max_j(min(xi, ij, jz))) = min(xi, max_j(min(ij, jz)))
					mmx = mcp::kernel::max_of_pairmin(matrix->data(i), row_z, count);   // TODO: use a mask to mask out multiple entries.
					mn = std::min(row_x[i], mmx);
					mx = std::max(mx, mn);

				}
			}
			return static_cast<IT>(mx);
		}
};


template <typename IT, bool ALLOW_LOOP=false>
class maxmin4_kernel; 
template <typename IT>
class maxmin4_kernel<IT, false> : public splash::kernel::inner_product_pos<IT, IT, splash::kernel::DEGREE::VECTOR, splash::kernel::DEGREE::VECTOR> {
	protected:
		::splash::ds::aligned_matrix<IT> const * matrix;
		int transition;
		std::vector<double> const * TFs;

	public:
		maxmin4_kernel(splash::ds::aligned_matrix<IT> const * mat = nullptr, int const & _transition = -1, std::vector<double> const * _TFs = nullptr) : matrix(mat), transition(_transition), TFs(_TFs) {}
		virtual ~maxmin4_kernel() {}

        void copy_parameters(maxmin4_kernel const & other) {
			matrix = other.matrix;
			transition = other.transition;
        	TFs = other.TFs;
        }

		inline virtual IT operator()(size_t const & x, size_t const & z, IT const * row_x, IT const * row_z, size_t const & count) const {
			// note x and z entries are set to 0.  (self MI excluded).  Then the max is taken.
			if (matrix == nullptr) return 0;
			if (x == z) return 0;

			// naive approach.  use a double loop for x-i-k-j-z, iterating over all r and s.
			// note, skip x, and z.
			double mx = std::numeric_limits<double>::lowest(); // for max over all i, j, of min(xi, ij, jz)
			double mn, mn2;  //  min(xi, ij, jz)
			if ((transition > 0) && (TFs != nullptr) && (TFs->size() >= count)) {
				// skip row_x[x], row_x[z], row_z[x], row_z[z]
				for (size_t i = 0; i < count; ++i) {
					// get the minimum along path x-i-k-j-z
					if ((i == x) || (i == z)) continue;
					if ((transition >= 1) && ((*TFs)[i] < 0)) continue;  // skip if not TF and ttggg tttgg ttttg

					for (size_t k = 0; k < count; ++k) {
						if (i == k) continue;
						if ((k == x) || (k == z)) continue;
						if ((transition >= 2) && ((*TFs)[k] < 0)) continue;  // skip if not TF and tttgg ttttg
							
						for (size_t j = 0; j < count; ++j) {
							if ((j == i) || (j == k)) continue;
							if ((j == x) || (j == z)) continue;
							if ((transition >= 3) && ((*TFs)[j] < 0)) continue;  // skip if not TF and ttttg

							mn = std::min(row_x[i], (*matrix)(i, k));
							mn = std::min(mn, (*matrix)(k, j));
							mn = std::min(mn, row_z[j]);

							// get the max of the match
							mx = std::max(mx, mn);
						}
					}
				}

			} else {  // 0 or less:  t-g-g-g-g or g-g-g-g-g
				IT xi, mmx;
				IT const * mati;
				std::set<size_t> excl;
				excl.insert(x);
				excl.insert(z);

				// skip row_x[x], row_x[z], row_z[x], row_z[z]
				for (size_t i = 0; i < count; ++i) {
					// get the minimum along path x-i-k-j-z
					if ((i == x) || (i == z)) continue;
					excl.insert(i);
					xi = row_x[i];
					mati = matrix->data(i);
					for (size_t k = 0; k < count; ++k) {
						if (i == k) continue;
						if ((k == x) || (k == z)) continue;
						excl.insert(k);
						mn2 = std::min(xi, row_z[k]);
						mmx = max_of_pairmin(mati, matrix->data(k), count, excl);
						mn = std::min(mmx, mn2);
						mx = std::max(mx, mn);
						
						// 	mn = std::min(row_x[i], row_z[k]);
						// for (size_t j = 0; j < count; ++j) {
						// 	if ((j == i) || (j == k)) continue;
						// 	if ((j == x) || (j == z)) continue;
							
						// 	mn2 = std::min((*matrix)(k, i), (*matrix)(k, j));
						// 	mn = std::min(mn, mn2);

						// 	// get the max of the match
						// 	mx = std::max(mx, mn);
						// }
						excl.erase(k);
					}
					excl.erase(i);
				}
			}
			return static_cast<IT>(mx);
		}
};


template <typename IT>
class maxmin4_kernel<IT, true> : public splash::kernel::inner_product<IT, IT, splash::kernel::DEGREE::VECTOR, splash::kernel::DEGREE::VECTOR> {
	protected:
		::splash::ds::aligned_matrix<IT> const * matrix;
		int transition;
		std::vector<double> const * TFs;

	public:
		maxmin4_kernel(splash::ds::aligned_matrix<IT> const * mat = nullptr, int const & _transition = -1, std::vector<double> const * _TFs = nullptr) : matrix(mat), transition(_transition), TFs(_TFs) {}
		virtual ~maxmin4_kernel() {}

        void copy_parameters(maxmin4_kernel const & other) {
			matrix = other.matrix;
			transition = other.transition;
        	TFs = other.TFs;
        }

		inline virtual IT operator()(IT const * row_x, IT const * row_z, size_t const & count) const {
			// note x and z entries are set to 0.  (self MI excluded).  Then the max is taken.
			if (matrix == nullptr) return 0;
			
			// naive approach.  use a double loop for x-i-k-j-z, iterating over all r and s.
			// note, skip x, and z.
			double mx = std::numeric_limits<double>::lowest(); // for max over all i, j, of min(xi, ij, jz)
			double mn, mn2;  //  min(xi, ij, jz)
			if ((transition > 0) && (TFs != nullptr) && (TFs->size() >= count)) {
				// skip row_x[x], row_x[z], row_z[x], row_z[z]
				for (size_t i = 0; i < count; ++i) {
					// get the minimum along path x-i-k-j-z
					if ((transition >= 1) && ((*TFs)[i] < 0)) continue;  // skip if not TF and ttggg tttgg ttttg

					for (size_t k = 0; k < count; ++k) {
						if ((transition >= 2) && ((*TFs)[k] < 0)) continue;  // skip if not TF and tttgg ttttg
							
						for (size_t j = 0; j < count; ++j) {
							if ((transition >= 3) && ((*TFs)[j] < 0)) continue;  // skip if not TF and ttttg

							mn = std::min(row_x[i], (*matrix)(i, k));
							mn = std::min(mn, (*matrix)(k, j));
							mn = std::min(mn, row_z[j]);

							// get the max of the match
							mx = std::max(mx, mn);
						}
					}
				}

			} else {  // 0 or less:  t-g-g-g-g or g-g-g-g-g
				IT xi, mmx;
				IT const * mati;
			
				// skip row_x[x], row_x[z], row_z[x], row_z[z]
				for (size_t i = 0; i < count; ++i) {
					// get the minimum along path x-i-k-j-z
					xi = row_x[i];
					mati = matrix->data(i);
					for (size_t k = 0; k < count; ++k) {
						mn2 = std::min(xi, row_z[k]);
						mmx = max_of_pairmin(mati, matrix->data(k), count);
						mn = std::min(mmx, mn2);
						mx = std::max(mx, mn);
						
					}
				}
			}
			return static_cast<IT>(mx);
		}
};


// apply one of the above using innerproduct pattern.   The output is a boolean matrix.
// then use threshold to apply boolean matrix to input matrix.


}}