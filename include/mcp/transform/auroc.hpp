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

#include "splash/kernel/kernel_base.hpp"
#include <vector>
#include <algorithm>

namespace wave { namespace kernel { 

// ROC:
// true positive rate (sensitivity) vs false positive rate
// sensitivity == recall == TP/P
// false positive rate == 1 - TN/N = FP / N
// above threshold is considered positive.  threshold from low to high = TPR from 1 to 0.  FPR also from 1 to 0

// conceptual approach:
//  1. sort {P, N} keeping label attached to each element.  walk through and track TP and FP counts. (FN follows)
//  2. accumulate via either average or trapezoid rule.  (common part)

// implementation:
//  1. sort P and sort N (avoid keeping labels).  walk through both lists, and use min of list heads as next threshold.

//  since both average and trapezoid rule depend on X axis interval, if a threshold change does not produce changes in X value, then the AUPR and AUROC increase is 0.
//  Thresholds that effect changes in X are the unique source values corresponding to X (TP/P for AUPR, FP/N for AUROC)
//  2. sort X source value (P for AUPR, N for AUROC), and walk through the unique values of those ONLY.
//     caveat: multiple X thresholds may result in the same Y.  this produces a flat curve.  this is okay.  example is threshold of 7 and 8 below:  Y rate = 5/6.
//     caveat: multiple Y thresholds result in the same X. e.g. Y = {4 5 6} all produce X rate = 1/3.  should not be an issue.
//            
//      e.g.  Y Y X Y Y Y X X Y   should look like (2/6, 5/6, 5/6, 1)
//            1 2 3 4 5 6 7 8 9
//  a. walking through sorted Y
//  b. binary search through Y

// 2b. is fastest.
// also, since X axis and Y axis are respectively FP/N and TP/P only,
//   we can swap X and Y and computer 1-auc.

// TODO:
// [ ] sort true edges and not negative edges.  use binary search for negatives. (about 50% faster)
// [ ] dynamic depending on size.
// [ ] faster than mlog(n)? 
// [ ] OMP parallel: auroc is a reduction of sorted data - can partition.  
// [ ] OMP parallel: but what about the sort or binary search methods - partition array?  how to aggregate?  parallel sort...
// [ ] MPI parallel: local sort, local counts, global reduce?

// NOTE: no FPR change means no auroc contribution.  this means no FP change.
//    but recall can change even if FPR stays the same.  this affects the next block of auroc when there IS FPR change.
//    for a block, want: precision before and after FP change (thus FPR change).   so need to know FP change
//   TP changes when true threshold changes, so each unique value in the true array, or we can say at the end of an equal range.
//   FP change that matters (recall changes), so first threshold > true value.
//   so each block:  left side is positive threshold.  right is first greater negative threshold
 
// compute AUROC
// uses trapezoid rule
// complexity is O(nlog(n)) because of the sorting operation
// ground truth as true (1), false (0), and unknown (-1)
// precision = tp/(tp+fp),  FPR = fp/(fp+tn)
// Based on sklearn.metrics, even when FPR does not change, we still need to compute the changes in recall
//     AUC parts is computed using the recall at the 2 ends of a FPR segment.
// this situation happens when
//     1. FP does not advance, so FPR does not change.
//     2. TP does change, so recall changes. specifically, we need TP at the first negative threshold that produces the next FP.
// the negative thresholds need to be ordered, so either binning, or sorting. binning ~= histogram. given non-uniform boundary, this is not linear.
//     eitherway, we are doing mlogm or mlogn.
template <typename IT, typename LABEL, typename OT>
class auroc_kernel : public splash::kernel::masked_reduce<IT, LABEL, OT, splash::kernel::DEGREE::VECTOR> {
    protected:
        mutable std::vector<IT> gold_pos;
        mutable std::vector<IT> gold_neg;
        LABEL const pos;
        LABEL const neg;

	public:
		using InputType = IT;
		using Input2Type = LABEL;
        using OutputType = OT;

		auroc_kernel(LABEL const & p = 1, LABEL const & n = 0) : pos(p), neg(n) {}
		virtual ~auroc_kernel() {};
        void copy_parameters(auroc_kernel const & other) {
            other.pos = pos;
            other.neg = neg;
        }

        inline virtual void initialize(size_t const & count) {
            gold_pos.reserve(count);
            gold_neg.reserve(count);
        }

    protected:
        // count the number of elements less than or equal to 
        // Y_lts: number of Y entries strictly less than each X thresholds (exclude equals)
        // Y_eqs: number of Y entries equal to X thresholds
        // NOTE: output are 2 arrays of size GX+1.
        // NOTE: if there are repeats in sorted_X, the position in the output corresponding to first of repeats
        //          is incremented, due to lower_bound.
        inline void count_le(std::vector<IT> const & sorted_X, std::vector<IT> const & Y, 
            std::vector<size_t> & Y_lts, std::vector<size_t> & Y_eqs) const {
            size_t GX = sorted_X.size();

            Y_lts.clear();
            Y_lts.resize(GX+1, 0);
            Y_eqs.clear();
            Y_eqs.resize(GX+1, 0);
            
            auto xit = sorted_X.begin();
            auto xend = sorted_X.end();
            auto xstart = sorted_X.begin();

            size_t i = 0;

            // count the number of elements for each X threshold
            for (IT y : Y) {  // for each y value
                xit = std::lower_bound(xstart, xend, y);  // finds y <= X.
                if ((xit == xend))  {
                    // X < y for all x.  increment the last entry.  note equal does not work here.
                    ++Y_lts[GX];
                } else {
                    i = std::distance(xstart, xit);  // get the position, range [0 N-1]
                    if (*xit == y) { // 
                        // store counts of y == X[i].  these are used to add to the Y values to skip over jumps in Y. (if we use the pre-jump value, the AUC is decreased.)
                        ++Y_eqs[i];
                    } else {
                        // Y < X[i].  note because of lower_bound, we have X[i-1] < y strictly
                        ++Y_lts[i];
                    }
                }
            }
        }


        // for TPR vs FPR curve. X axis follows N, and Y axis follows P.
        // depending on which is smaller, we should sort that.  If P is smaller, then switch axis, effectively compute area above curve, then do 1-A
        inline virtual OT average_compute_1sort(std::vector<IT> const & YV, std::vector<IT> & XV) const {
            FMT_ROOT_PRINT("AVG, 1SORT, sizes:  pos {}, neg {}\n", YV.size(), XV.size());
            
            if (XV.size() == 0) return 0;  // no X, interval support is 0, so zero.
            if (YV.size() == 0) return 0;  // no Y, p is 0, so 

            size_t nP = YV.size();  // gold positive.
            size_t nN = XV.size();  // gold negative

            // linear scan XV and binary search YV is not the way to go because 
            //  unordered XV does not provide count info.
            
            std::vector<size_t> y_less;
            std::vector<size_t> y_equal;

            // begin with assumption that nN is smaller.

            // sort positive list in ascending order.
            auto stime = getSysTime();
            std::sort(XV.begin(), XV.end());   // nlogn
            auto etime = getSysTime();
            FMT_ROOT_PRINT("Sort pos elapsed {}\n", get_duration_s(stime, etime));

            stime = getSysTime();
            // only first of a repeat stretch has non-zero value sin y_less and y_equal.
            count_le(XV, YV, y_less, y_equal);
            etime = getSysTime();
            FMT_ROOT_PRINT("Sort neg elapsed {}\n", get_duration_s(stime, etime));
            // if we search pos threshold in the negative vector, we have nlogm, but overall is still mlogm dominated.
            // linear write here.

            // initial:  threshold is below min value, so all values are predicted as positive:  TP and FP are full size.
            size_t TP = nP;  // part of YV above threshold
            size_t FP = nN;  // part of XV above threshold
            // FN not needed as denominator for recall is nP = TP + FN (gold pos).
            // size_t FN = 0;  // part of YV above threshold

            double last_y;  // threshold below min value.
            // double prec;   // starting position for precision is 1.0. (0/0.  define as 1.)
            // double last_recall = 1.0, recall;  // starting at TP = nP, and FN = 0.  so value is 1.
            double auroc = 0.0;
            double y_inv_denom = 1.0 / static_cast<double>(nP);
            double x_inv_denom = 1.0 / static_cast<double>(nN);
            size_t last_x;

            // PR Curve (and ROC) evaluates P and R for a set of thresholds.
            // thresholds may be equally sampled or adaptive.
            // however, if consecutive thresholds do not change TP or FP,
            // then the curve would be flat for that part.
            // more specifically, if R, i.e. TP, does not change, even if P changes due to FP,
            //  there is no contribution to AUROC as the trapezoid has no height.

            // for TP to change, we only need to inspect values in the YV array.
            
            // first value
            IT thresh = XV[0];
            // double rec;
            size_t i = 0;
            TP = nP;

            // method:
            // incrementally walk through each value in the YV lists  (ascending order)
            // this means that TP and FP are decreasing, and FN is increasing.
            // calculate precision and recall from TP, FP, and FN.
            // note that TP and FP changes for ech YV threshold exceeded.
            for (; (i < nN); ) {
                // scan over all the equivalues in each list
                // NOTE: in skm learn, TP is from "val >= threshold"
                while ((i < nN) && (XV[i] < thresh)) ++i;

                TP -= y_less[i];

                // FMT_ROOT_PRINT("last thresh: i, TP, FP: {}, {}, {}, {}\n", thresh, 
                //     i, TP, FP);

                // TP increased, so recall increased.
                last_y = static_cast<double>(TP) * y_inv_denom;
                
                // rec = static_cast<double>(TP) / static_cast<double>(nP);
                last_x = nN - i;
                // FMT_ROOT_PRINT("last p/r: {}/{}\n", last_prec, rec);

                // find the next threshold.  either will change TP.  maybe change FP.
                TP -= y_equal[i];
                while ((i < nN) && (XV[i] <= thresh)) ++i;
                FP = nN - i;

                // FMT_ROOT_PRINT("thresh: i, TP, FP: {}, {}, {}, {}\n", thresh, 
                //     i, TP, FP);

                auroc += last_y * (static_cast<double>(last_x - FP) * x_inv_denom);
                // now get past the equal stretch so we can find the next thresh
                if (i < nN) {
                    thresh = XV[i];
                }

            }

            // final bit, TP becomes zero, so no contribution.

            // return auroc.
            return auroc;
        }


        inline virtual OT trapezoid_compute_1sort(std::vector<IT> const & YV, std::vector<IT> & XV) const {
            FMT_ROOT_PRINT("TRAP, 1SORT, sizes:  pos {}, neg {}\n", YV.size(), XV.size());
            
            if (YV.size() == 0) return 0;
            if (XV.size() == 0) return 0;

            size_t nP = YV.size();  // gold positive.
            size_t nN = XV.size();  // gold negative

            // linear scan XV and binary search YV is not the way to go because 
            //  unordered XV does not provide count info.
            
            std::vector<size_t> y_less;
            std::vector<size_t> y_equal;

            // sort positive list in ascending order.
            auto stime = getSysTime();
            std::sort(XV.begin(), XV.end());   // nlogn
            auto etime = getSysTime();
            FMT_ROOT_PRINT("Sort pos elapsed {}\n", get_duration_s(stime, etime));

            stime = getSysTime();
            count_le(XV, YV, y_less, y_equal);
            etime = getSysTime();
            FMT_ROOT_PRINT("Sort neg elapsed {}\n", get_duration_s(stime, etime));
            // if we search pos threshold in the negative vector, we have nlogm, but overall is still mlogm dominated.
            // linear write here.

            // initial:  threshold is below min value, so all values are predicted as positive:  TP and FP are full size.
            size_t TP = nP;  // part of YV above threshold
            size_t FP = nN;  // part of XV above threshold
            // FN not needed as denominator for recall is nP = TP + FN (gold pos).
            // size_t FN = 0;  // part of YV above threshold

            double last_y;  // threshold below min value.
            double y;   // starting position for precision is 1.0. (0/0.  define as 1.)
            // double last_recall = 1.0, recall;  // starting at TP = nP, and FN = 0.  so value is 1.
            double auroc = 0.0;
            double y_inv_denom = 1.0 / static_cast<double>(nP);
            double x_inv_denom = 1.0 / static_cast<double>(nN);
            size_t last_x;

            // PR Curve (and ROC) evaluates P and R for a set of thresholds.
            // thresholds may be equally sampled or adaptive.
            // however, if consecutive thresholds do not change TP or FP,
            // then the curve would be flat for that part.
            // more specifically, if R, i.e. TP, does not change, even if P changes due to FP,
            //  there is no contribution to AUROC as the trapezoid has no height.

            // for TP to change, we only need to inspect values in the YV array.
            
            // first value
            IT thresh = XV[0];
            // double rec;
            size_t i = 0;
            TP = nP;

            // method:
            // incrementally walk through each value in the YV lists  (ascending order)
            // this means that TP and FP are decreasing, and FN is increasing.
            // calculate precision and recall from TP, FP, and FN.
            // note that TP and FP changes for ech YV threshold exceeded.
            for (; (i < nN); ) {
                // scan over all the equivalues in each list
                // NOTE: in skm learn, TP in AUPR is from "val >= threshold"
                while ((i < nN) && (XV[i] < thresh)) ++i;
                                            
                TP -= y_less[i];

                // FMT_ROOT_PRINT("last thresh: i, TP, FP: {}, {}, {}, {}\n", thresh, 
                //     i, TP, FP);


                // TP increased, so recall increased.
                last_y = static_cast<double>(TP) * y_inv_denom;

                // rec = static_cast<double>(TP) / static_cast<double>(nP);
                last_x = nN - i;
                // FMT_ROOT_PRINT("last p/r: {}/{}\n", last_prec, rec);

                // find the next threshold.  either will change TP.  maybe change FP.
                TP -= y_equal[i];
                while ((i < nN) && (XV[i] <= thresh)) ++i;
                FP = nN - i;  // spacing between last_x and FP is either 1 or larger if there equal elements.

                // FMT_ROOT_PRINT("thresh: i, j, TP, FP: {}, {}, {}, {}\n", thresh, 
                //     i, TP, FP);

                // trapezoid rule:
                y = static_cast<double>(TP) * y_inv_denom;

                auroc += (y + last_y) * 0.5 * (static_cast<double>(last_x - FP) * x_inv_denom);
                // now get past the equal stretch so we can find the next thresh
                if (i < nN) {
                    thresh = XV[i];
                }

            }
            // final bit, y_less[nN] maybe >0, y_equal[nN] == 0. TP is zero for both, so no additional contribution.

            // return auroc.
            return auroc;
        }



    public:
        inline virtual OT compute_full_mat(splash::ds::aligned_matrix<IT> const & in, splash::ds::aligned_matrix<LABEL> const & labels) {
            assert((in.rows() == labels.rows()) && "input and label sizes must be same.\n");
            assert((in.columns() == labels.columns()) && "input and label sizes must be same.\n");

            // scan and split into two lists
            auto stime = getSysTime();
            gold_pos.clear();
            gold_neg.clear();
            IT const * inptr;
            LABEL const * labptr;
            for (size_t i = 0; i < in.rows(); ++i) {
                inptr = in.data(i);
                labptr = labels.data(i);
                for (size_t j = 0; j < in.columns(); ++j, ++inptr, ++labptr) {
                    if (*labptr == pos) {
                        gold_pos.emplace_back(*inptr);
                    } else if (*labptr == neg) {
                        gold_neg.emplace_back(*inptr);
                    }
                }
            }
            auto etime = getSysTime();
            FMT_ROOT_PRINT("prep for AUROC {}\n", get_duration_s(stime, etime));

            if (gold_pos.size() < gold_neg.size()) {
                return 1 - average_compute_1sort(gold_neg, gold_pos);
            } else {
                return average_compute_1sort(gold_pos, gold_neg);
            }
            // return test_and_bench(gold_pos, gold_neg);
        }
        inline virtual OT compute_no_diagonal(splash::ds::aligned_matrix<IT> const & in, splash::ds::aligned_matrix<LABEL> const & labels) {
            assert((in.rows() == labels.rows()) && "input and label sizes must be same.\n");
            assert((in.columns() == labels.columns()) && "input and label sizes must be same.\n");

            // scan and split into two lists
            auto stime = getSysTime();
            gold_pos.clear();
            gold_neg.clear();
            IT const * inptr;
            LABEL const * labptr;
            for (size_t i = 0; i < in.rows(); ++i) {
                inptr = in.data(i);
                labptr = labels.data(i);
                for (size_t j = 0; j < in.columns(); ++j, ++inptr, ++labptr) {
                    if (i == j) continue;
                    if (*labptr == pos) {
                        gold_pos.emplace_back(*inptr);
                    } else if (*labptr == neg) {
                        gold_neg.emplace_back(*inptr);
                    }
                }
            }
            auto etime = getSysTime();
            FMT_ROOT_PRINT("prep for AUROC {}\n", get_duration_s(stime, etime));

            if (gold_pos.size() < gold_neg.size()) {
                return 1 - average_compute_1sort(gold_neg, gold_pos);
            } else {
                return average_compute_1sort(gold_pos, gold_neg);
            }
            // return test_and_bench(gold_pos, gold_neg);

        }
        inline virtual OT compute_upper_triangle(splash::ds::aligned_matrix<IT> const & in, splash::ds::aligned_matrix<LABEL> const & labels) {
            assert((in.rows() == labels.rows()) && "input and label sizes must be same.\n");
            assert((in.columns() == labels.columns()) && "input and label sizes must be same.\n");

            // scan and split into two lists
            auto stime = getSysTime();
            gold_pos.clear();
            gold_neg.clear();
            IT const * inptr;
            LABEL const * labptr;
            for (size_t i = 0; i < in.rows(); ++i) {
                inptr = in.data(i, i+1);
                labptr = labels.data(i, i+1);
                for (size_t j = i+1; j < in.columns(); ++j, ++inptr, ++labptr) {
                    if (*labptr == pos) {
                        gold_pos.emplace_back(*inptr);
                    } else if (*labptr == neg) {
                        gold_neg.emplace_back(*inptr);
                    }
                }
            }
            auto etime = getSysTime();
            FMT_ROOT_PRINT("prep for AUROC {}\n", get_duration_s(stime, etime));

            if (gold_pos.size() < gold_neg.size()) {
                return 1 - average_compute_1sort(gold_neg, gold_pos);
            } else {
                return average_compute_1sort(gold_pos, gold_neg);
            }
            // return test_and_bench(gold_pos, gold_neg);
        }
        inline virtual OT compute_lower_triangle(splash::ds::aligned_matrix<IT> const & in, splash::ds::aligned_matrix<LABEL> const & labels) {
            assert((in.rows() == labels.rows()) && "input and label sizes must be same.\n");
            assert((in.columns() == labels.columns()) && "input and label sizes must be same.\n");

            // scan and split into two lists
            auto stime = getSysTime();
            gold_pos.clear();
            gold_neg.clear();
            IT const * inptr;
            LABEL const * labptr;
            size_t last;
            for (size_t i = 0; i < in.rows(); ++i) {
                inptr = in.data(i);
                labptr = labels.data(i);
                last = std::min(i, in.columns());
                for (size_t j = 0; j < last; ++j, ++inptr, ++labptr) {
                    if (*labptr == pos) {
                        gold_pos.emplace_back(*inptr);
                    } else if (*labptr == neg) {
                        gold_neg.emplace_back(*inptr);
                    }
                }
            }
            auto etime = getSysTime();
            FMT_ROOT_PRINT("prep for AUROC {}\n", get_duration_s(stime, etime));

            if (gold_pos.size() < gold_neg.size()) {
                return 1 - average_compute_1sort(gold_neg, gold_pos);
            } else {
                return average_compute_1sort(gold_pos, gold_neg);
            }
            // return test_and_bench(gold_pos, gold_neg);
        }

        inline virtual OT operator()(splash::ds::aligned_vector<IT> const & in, splash::ds::aligned_vector<LABEL> const & labels) {
            assert((in.size() == labels.size()) && "input nad label sizes must be same.\n");

            return this->operator()(in.data(), labels.data(), in.size());
        }

        inline virtual OT operator()(IT const * in, LABEL const * labels, size_t const & count) const {

            // scan and split into two lists
            auto stime = getSysTime();
            gold_pos.clear();
            gold_neg.clear();
            LABEL const * lb = labels;
            IT const * inn = in;
            for (size_t i = 0; i < count; ++i, ++lb, ++inn) {
                if (*lb == pos) {
                    gold_pos.emplace_back(*inn);
                } else if (*lb == neg) {
                    gold_neg.emplace_back(*inn);
                }
            }
            auto etime = getSysTime();
            FMT_ROOT_PRINT("prep for AUROC {}\n", get_duration_s(stime, etime));

            if (gold_pos.size() < gold_neg.size()) {
                return 1 - average_compute_1sort(gold_neg, gold_pos);
            } else {
                return average_compute_1sort(gold_pos, gold_neg);
            }
            // return test_and_bench(gold_pos, gold_neg);
        }
#ifdef USE_MPI
        inline virtual OT operator()(splash::ds::aligned_vector<IT> const & pos,
            splash::ds::aligned_vector<IT> const & neg, MPI_Comm comm = MPI_COMM_WORLD) const {

            int rank;
            int procs;
            MPI_Comm_rank(comm, &rank);
            MPI_Comm_size(comm, &procs);

            auto stime = getSysTime();
            // gather

            auto all_pos = pos.gather(0, comm);
            auto all_neg = neg.gather(0, comm);
            auto etime = getSysTime();
            FMT_ROOT_PRINT("gather for AUROC in {} sec\n", get_duration_s(stime, etime));

            stime = getSysTime();   
            OT auroc = 0;
            if (rank == 0) {
                auroc = this->operator()(all_pos.data(), all_pos.size(), all_neg.data(), all_neg.size());
            } 
            etime = getSysTime();
            FMT_ROOT_PRINT("compute for AUROC {} in {} sec\n", auroc, get_duration_s(stime, etime));

            // then scatter results.
            splash::utils::mpi::datatype<OT> dt;
            MPI_Bcast(&auroc, 1, dt.value, 0, comm);

            return auroc;
        }

#else
        inline virtual OT operator()(splash::ds::aligned_vector<IT> const & pos,
            splash::ds::aligned_vector<IT> const & neg) const {
            return this->operator()(pos.data(), pos.size(), neg.data(), neg.size());
        }
#endif

        inline virtual OT operator()(IT const * pos, size_t const & pos_count,
             IT const * neg, size_t const & neg_count) const {

            // scan and split into two lists
            auto stime = getSysTime();
            gold_pos.resize(pos_count);
            memcpy(gold_pos.data(), pos, pos_count * sizeof(IT));
            gold_neg.resize(neg_count);
            memcpy(gold_neg.data(), neg, neg_count * sizeof(IT));
            
            auto etime = getSysTime();
            FMT_ROOT_PRINT("prep for AUROC in {} sec\n", get_duration_s(stime, etime));

            if (gold_pos.size() < gold_neg.size()) {
                return 1 - average_compute_1sort(gold_neg, gold_pos);
            } else {
                return average_compute_1sort(gold_pos, gold_neg);
            }
            // return test_and_bench(gold_pos, gold_neg);
        }
}; 


}}