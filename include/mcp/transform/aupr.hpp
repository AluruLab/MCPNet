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

namespace mcp { namespace kernel { 

// TODO:
// [X] sort true edges and not negative edges.  use binary search for negatives. (about 50% faster)
// [X] dynamic depending on size.
// [ ] faster than mlog(n)? 
// [ ] OMP parallel: aupr is a reduction of sorted data - can partition.  
// [ ] OMP parallel: but what about the sort or binary search methods - partition array?  how to aggregate?  parallel sort...
// [ ] MPI parallel: local sort, local counts, global reduce?

// NOTE: no recall change means no aupr contribution.  this means no tp change.
//    but precision can change even if recall stays the same.  this affects the next block of aupr when there IS recall change.
//    for a block, want: precision before and after TP change (thus recall change).   so need to know FP change
//   TP changes when true threshold changes, so each unique value in the true array, or we can say at the end of an equal range.
//   FP change that matters (recall changes), so first threshold > true value.
//   so each block:  left side is positive threshold.  right is first greater negative threshold

// ROC (TPR vs FPR) or its inverse (FPR vs TPR) are both functions, so can compute by switching axes, and sorting by the smaller list.
// PR curve is a function, but its inverse may not be because of TP/(TP+FP), which conflates both P and N, and may not behave monotonically.  Cannot use the same trick as ROC.
 
// compute AUPR
// uses trapezoid rule
// complexity is O(nlog(n)) because of the sorting operation
// ground truth as true (1), false (0), and unknown (-1)
// precision = tp/(tp+fp),  recall = tp/(tp+fn)
// Based on sklearn.metrics, even when recall does not change, we still need to compute the changes in precision
//     AUC parts is computed using the precisions at the 2 ends of a recall segment.
// this situation happens when
//     1. TP does not advance, so recall does not change.
//     2. FP does change, so precision changes. specifically, we need FP at the first negative threshold that produces the next TP.
// the negative thresholds need to be ordered, so either binning, or sorting. binning ~= histogram. given non-uniform boundary, this is not linear.
//     eitherway, we are doing mlogm or mlogn.
template <typename IT, typename LABEL, typename OT>
class aupr_kernel : public splash::kernel::masked_reduce<IT, LABEL, OT, splash::kernel::DEGREE::VECTOR> {
    protected:
        mutable std::vector<IT> gold_pos;
        mutable std::vector<IT> gold_neg;
        LABEL const pos;
        LABEL const neg;

	public:
		using InputType = IT;
		using Input2Type = LABEL;
        using OutputType = OT;

		aupr_kernel(LABEL const & p = 1, LABEL const & n = 0) : pos(p), neg(n) {}
		virtual ~aupr_kernel() {};
        void copy_parameters(aupr_kernel const & other) {
            other.pos = pos;
            other.neg = neg;
        }

        inline virtual void initialize(size_t const & count) {
            gold_pos.reserve(count);
            gold_neg.reserve(count);
        }

    protected:
        // neg_lts: number of negative entries less than the positive thresholds (exclude equals)
        // neg_eqs: number of negative entries equal to positive thresholds
        // note that output is are 2 arrays of size GP+1. if in gold_pos there are repeats, 
        // the first of equal elements is incremented due to "lower_bound"
        inline virtual void ordering(std::vector<size_t> & neg_lts, std::vector<size_t> & neg_eqs) const {
            size_t GP = gold_pos.size();

            neg_lts.clear();
            // for each positive threshold, store the number of negative values less than.
            neg_lts.resize(GP+1, 0);
            neg_eqs.clear();
            // for each negative threshold in next_thresh, store the number of negative thresholds less than.
            // given next_thresh definition (next neg value greater than pos), this is same as 
            // for each positive threshold, store the number of negative thresholds less than or equal to pos_thresh.
            // probably easier just to count the equal to pos_thresh.
            neg_eqs.resize(GP+1, 0);
            
            auto pit = gold_pos.begin();
            auto pend = gold_pos.end();
            auto pstart = gold_pos.begin();

            size_t i = 0;

            // looking for next positive threshold greater than the neg.  equal case would be encompassed by the positive threshold.
            for (IT neg_thresh : gold_neg) {
                // search for lower bound
                pit = std::lower_bound(pstart, pend, neg_thresh);  // neg <= pos_{n}.
                if (pit == pend) {
                    ++neg_lts[GP];
                } else {
                    i = std::distance(pstart, pit);
                    if (*pit == neg_thresh) {
                    // store counts of negatives equal to positives.  once the next smallest neg > pos is found, these are its FP additions.
                        ++neg_eqs[i];
                    } else {
                        // neg < pos_{n}.  note because of lower_bound, we have pos_{n-1} < neg strictly
                        // store counts of negative smaller than positive.
                        ++neg_lts[i];
                    }
                }
            }

        }

        inline virtual OT average_compute_1sort() const {
            FMT_ROOT_PRINT("AVG, 1SORT, sizes:  pos {}, neg {}\n", gold_pos.size(), gold_neg.size());
            
            if (gold_pos.size() == 0) return 0;

            size_t GP = gold_pos.size();  // gold positive.
            size_t GN = gold_neg.size();  // gold negative

            // linear scan gold_neg and binary search gold_pos is not the way to go because 
            //  unordered gold_neg does not provide count info.
            
            // sort positive list in ascending order.
            auto stime = getSysTime();
            std::sort(gold_pos.begin(), gold_pos.end());   // nlogn
            auto etime = getSysTime();
            FMT_ROOT_PRINT("Sort pos elapsed {}\n", get_duration_s(stime, etime));

            stime = getSysTime();
            std::vector<size_t> neg_less;
            std::vector<size_t> neg_equal;

            ordering(neg_less, neg_equal);
            etime = getSysTime();
            FMT_ROOT_PRINT("Sort neg elapsed {}\n", get_duration_s(stime, etime));
            // if we search pos threshold in the negative vector, we have nlogm, but overall is still mlogm dominated.
            // linear write here.

            // for (auto x : gold_pos) {
            //     FMT_ROOT_PRINT("pos {}\n", x);
            // }
            // for (auto x : gold_neg) {
            //     FMT_ROOT_PRINT("neg {}\n", x);
            // }

            // initial:  threshold is below min value, so all values are predicted as positive:  TP and FP are full size.
            size_t TP = GP;  // part of gold_pos above threshold
            size_t FP = GN;  // part of gold_neg above threshold
            // FN not needed as denominator for recall is GP = TP + FN (gold pos).
            // size_t FN = 0;  // part of gold_pos above threshold

            double last_prec;  // threshold below min value.
            // double prec;   // starting position for precision is 1.0. (0/0.  define as 1.)
            // double last_recall = 1.0, recall;  // starting at TP = GP, and FN = 0.  so value is 1.
            double aupr = 0.0;
            double inv_denom = 1.0 / static_cast<double>(GP);
            size_t last_TP;

            // PR Curve (and ROC) evaluates P and R for a set of thresholds.
            // thresholds may be equally sampled or adaptive.
            // however, if consecutive thresholds do not change TP or FP,
            // then the curve would be flat for that part.
            // more specifically, if R, i.e. TP, does not change, even if P changes due to FP,
            //  there is no contribution to AUPR as the trapezoid has no height.

            // for TP to change, we only need to inspect values in the gold_pos array.
            
            // first value
            IT thresh = gold_pos[0];
            // double rec;
            size_t i = 0;
            FP = GN;

            // method:
            // incrementally walk through each value in the gold_pos lists  (ascending order)
            // this means that TP and FP are decreasing, and FN is increasing.
            // calculate precision and recall from TP, FP, and FN.
            // note that TP and FP changes for ech gold_pos threshold exceeded.
            for (; (i < GP); ) {
                // scan over all the equivalues in each list
                // NOTE: in skm learn, TP is from "val >= threshold"
                while ((i < GP) && (gold_pos[i] < thresh)) ++i;
                                            
                TP = GP - i;
                FP -= neg_less[i];

                // FMT_ROOT_PRINT("last thresh: i, TP, FP: {}, {}, {}, {}\n", thresh, 
                //     i, TP, FP);


                // need to compute the start and end precision for the interval.
                // and the start and end recall for the interval for a threshold change.
                // thresholds are unique values in the gold_pos.
                // start and end are at the lower and upper bounds, (upper bound != last equal element)
                // recall = TP/GP then has an interval = (last_TP - TP) / GP
                if ((TP > 0) || (FP > 0) ) {
                    last_prec = static_cast<double>(TP) / static_cast<double>(TP + FP);
                } else {
                    last_prec = 1.0;
                }
                // rec = static_cast<double>(TP) / static_cast<double>(GP);
                last_TP = TP;
                // FMT_ROOT_PRINT("last p/r: {}/{}\n", last_prec, rec);

                // find the next threshold.  either will change TP.  maybe change FP.
                FP -= neg_equal[i];
                while ((i < GP) && (gold_pos[i] <= thresh)) ++i;
                TP = GP - i;

                // trapezoid rule:
                aupr += last_prec * (static_cast<double>(last_TP - TP) * inv_denom);
                // now get past the equal stretch so we can find the next thresh
                if (i < GP) {
                    thresh = gold_pos[i];
                }

            }

            // return aupr.
            return aupr;
        }


        inline virtual OT trapezoid_compute_1sort() const {
            FMT_ROOT_PRINT("TRAP, 1SORT, sizes:  pos {}, neg {}\n", gold_pos.size(), gold_neg.size());
            
            if (gold_pos.size() == 0) return 0;

            size_t GP = gold_pos.size();  // gold positive.
            size_t GN = gold_neg.size();  // gold negative

            // linear scan gold_neg and binary search gold_pos is not the way to go because 
            //  unordered gold_neg does not provide count info.
            
            // sort positive list in ascending order.
            auto stime = getSysTime();
            std::sort(gold_pos.begin(), gold_pos.end());   // nlogn
            auto etime = getSysTime();
            FMT_ROOT_PRINT("Sort pos elapsed {}\n", get_duration_s(stime, etime));

            stime = getSysTime();
            std::vector<size_t> neg_less;
            std::vector<size_t> neg_equal;

            ordering(neg_less, neg_equal);
            etime = getSysTime();
            FMT_ROOT_PRINT("Sort neg elapsed {}\n", get_duration_s(stime, etime));
            // if we search pos threshold in the negative vector, we have nlogm, but overall is still mlogm dominated.
            // linear write here.

            // for (auto x : gold_pos) {
            //     FMT_ROOT_PRINT("pos {}\n", x);
            // }
            // for (auto x : gold_neg) {
            //     FMT_ROOT_PRINT("neg {}\n", x);
            // }

            // initial:  threshold is below min value, so all values are predicted as positive:  TP and FP are full size.
            size_t TP = GP;  // part of gold_pos above threshold
            size_t FP = GN;  // part of gold_neg above threshold
            // FN not needed as denominator for recall is GP = TP + FN (gold pos).
            // size_t FN = 0;  // part of gold_pos above threshold

            double last_prec;  // threshold below min value.
            double prec;   // starting position for precision is 1.0. (0/0.  define as 1.)
            // double last_recall = 1.0, recall;  // starting at TP = GP, and FN = 0.  so value is 1.
            double aupr = 0.0;
            double inv_denom = 0.5 / static_cast<double>(GP);
            size_t last_TP;

            // PR Curve (and ROC) evaluates P and R for a set of thresholds.
            // thresholds may be equally sampled or adaptive.
            // however, if consecutive thresholds do not change TP or FP,
            // then the curve would be flat for that part.
            // more specifically, if R, i.e. TP, does not change, even if P changes due to FP,
            //  there is no contribution to AUPR as the trapezoid has no height.

            // for TP to change, we only need to inspect values in the gold_pos array.
            
            // first value
            IT thresh = gold_pos[0];
            // double rec;
            size_t i = 0;
            FP = GN;

            // method:
            // incrementally walk through each value in the gold_pos lists  (ascending order)
            // this means that TP and FP are decreasing, and FN is increasing.
            // calculate precision and recall from TP, FP, and FN.
            // note that TP and FP changes for ech gold_pos threshold exceeded.
            for (; (i < GP); ) {
                // scan over all the equivalues in each list
                // NOTE: in skm learn, TP is from "val >= threshold"
                while ((i < GP) && (gold_pos[i] < thresh)) ++i;
                                            
                TP = GP - i;
                FP -= neg_less[i];

                // FMT_ROOT_PRINT("last thresh: i, TP, FP: {}, {}, {}, {}\n", thresh, 
                //     i, TP, FP);


                // TP increased, so recall increased.
                if ((TP > 0) || (FP > 0) ) {
                    last_prec = static_cast<double>(TP) / static_cast<double>(TP + FP);
                } else {
                    last_prec = 1.0;
                }
                // rec = static_cast<double>(TP) / static_cast<double>(GP);
                last_TP = TP;
                // FMT_ROOT_PRINT("last p/r: {}/{}\n", last_prec, rec);

                // find the next threshold.  either will change TP.  maybe change FP.
                FP -= neg_equal[i];
                while ((i < GP) && (gold_pos[i] <= thresh)) ++i;
                TP = GP - i;

                // FMT_ROOT_PRINT("thresh: i, j, TP, FP: {}, {}, {}, {}\n", thresh, 
                //     i, TP, FP);

                // trapezoid rule:
                if ((TP > 0) || (FP > 0) ) {
                    prec = static_cast<double>(TP) / static_cast<double>(TP + FP);
                    // rec = static_cast<double>(TP) / static_cast<double>(GP);
                    // FMT_ROOT_PRINT("p/r: {}/{}\n", prec, rec);
                } else {
                    prec = 1.0;
                }
                aupr += (prec + last_prec) * (static_cast<double>(last_TP - TP) * inv_denom);                // now get past the equal stretch so we can find the next thresh
                if (i < GP) {
                    thresh = gold_pos[i];
                }

            }

            // return aupr.
            return aupr;
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
            FMT_ROOT_PRINT("prep for AUPR {}\n", get_duration_s(stime, etime));


            // n logn + m logn + n R(n) for 1sort.  R(n) is random mem access for n array
            // nlogn + mlogm + n + m for 1 thresh, 
            // if n is smaller, then 1sort wins.  
            // if m is smaller, then 1 thresh should win.
            // however, empirical result seems to suggest comparable performance. 
            return average_compute_1sort();
            // return test_and_bench();
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
            FMT_ROOT_PRINT("prep for AUPR {}\n", get_duration_s(stime, etime));


            // n logn + m logn + n R(n) for 1sort.  R(n) is random mem access for n array
            // nlogn + mlogm + n + m for 1 thresh, 
            // if n is smaller, then 1sort wins.  
            // if m is smaller, then 1 thresh should win.
            // however, empirical result seems to suggest comparable performance. 
            return average_compute_1sort();
            // return test_and_bench();

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
            FMT_ROOT_PRINT("prep for AUPR {}\n", get_duration_s(stime, etime));

            // n logn + m logn + n R(n) for 1sort.  R(n) is random mem access for n array
            // nlogn + mlogm + n + m for 1 thresh, 
            // if n is smaller, then 1sort wins.  
            // if m is smaller, then 1 thresh should win.
            // however, empirical result seems to suggest comparable performance. 
            return average_compute_1sort();
            // return test_and_bench();

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
            FMT_ROOT_PRINT("prep for AUPR {}\n", get_duration_s(stime, etime));

            // n logn + m logn + n R(n) for 1sort.  R(n) is random mem access for n array
            // nlogn + mlogm + n + m for 1 thresh, 
            // if n is smaller, then 1sort wins.  
            // if m is smaller, then 1 thresh should win.
            // however, empirical result seems to suggest comparable performance. 
            return average_compute_1sort();
            // return test_and_bench();

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
            FMT_ROOT_PRINT("prep for AUPR {}\n", get_duration_s(stime, etime));
            
            // n logn + m logn + n R(n) for 1sort.  R(n) is random mem access for n array
            // nlogn + mlogm + n + m for 1 thresh, 
            // if n is smaller, then 1sort wins.  
            // if m is smaller, then 1 thresh should win.
            // however, empirical result seems to suggest comparable performance. 
            return average_compute_1sort();
            // return test_and_bench();

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
            FMT_ROOT_PRINT("gather for AUPR in {} sec\n", get_duration_s(stime, etime));

            stime = getSysTime();   
            OT aupr = 0;
            if (rank == 0) {
                aupr = this->operator()(all_pos.data(), all_pos.size(), all_neg.data(), all_neg.size());
            } 

            // then scatter results.
            splash::utils::mpi::datatype<OT> dt;
            MPI_Bcast(&aupr, 1, dt.value, 0, comm);
            etime = getSysTime();
            FMT_ROOT_PRINT("compute for AUPR {} in {} sec\n", aupr, get_duration_s(stime, etime));

            return aupr;
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
            FMT_ROOT_PRINT("prep for AUPR in {} sec\n", get_duration_s(stime, etime));
            
            // n logn + m logn + n R(n) for 1sort.  R(n) is random mem access for n array
            // nlogn + mlogm + n + m for 1 thresh, 
            // if n is smaller, then 1sort wins.  
            // if m is smaller, then 1 thresh should win.
            // however, empirical result seems to suggest comparable performance. 
            return average_compute_1sort();
            // return test_and_bench();

        }
}; 


}}