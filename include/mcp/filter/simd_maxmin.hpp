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

#include <x86intrin.h>
#include <limits>
#include <algorithm>
#include <set>


namespace mcp { namespace kernel { 

// mask:  0 excludes  1 includes.
template<typename T>
inline T max_of_pairmin_scalar(T const * xx, T const * yy, size_t const & count, T const * mask = nullptr) {
    T val = std::numeric_limits<double>::lowest();

    if (mask) {
        // last ones (up to 3) if there.
        for (size_t i = 0; i < count; ++i) {
            val = std::max(val, std::min(mask[i], std::min(xx[i], yy[i])));
        }
    } else {
        // last ones (up to 3) if there.
        for (size_t i = 0; i < count; ++i) {
            val = std::max(val, std::min(xx[i], yy[i]));
        }
    }
    return val;
}


// compute the maximum of pairwise min using SSE (2 doubles per vec).  allow 1 exclusions in each array.
inline double max_of_pairmin_sse(double const * xx, double const * yy, size_t const & count, double const * mask = nullptr) {
#ifdef __SSE2__

    double val[2];

    __m128d a, b, c; 
    __m128d acc1 = _mm_set1_pd(std::numeric_limits<double>::lowest());
    __m128d acc2 = acc1;
    __m128d acc3 = acc1;
    __m128d acc4 = acc1;

    size_t max = count & 0xFFFFFFFFFFFFFFF8;
    size_t k = 0;
    if (mask) {
        // compute the bulk
        for (; k < max; k += 8) {
            a = _mm_loadu_pd(xx + k);
            b = _mm_loadu_pd(yy + k);
            c = _mm_loadu_pd(mask + k); 
            acc1 = _mm_max_pd(acc1, _mm_min_pd(c, _mm_min_pd(a, b)));

            a = _mm_loadu_pd(xx + k + 2);
            b = _mm_loadu_pd(yy + k + 2);
            c = _mm_loadu_pd(mask + k + 2); 
            acc2 = _mm_max_pd(acc2, _mm_min_pd(c, _mm_min_pd(a, b)));

            a = _mm_loadu_pd(xx + k + 4);
            b = _mm_loadu_pd(yy + k + 4);
            c = _mm_loadu_pd(mask + k + 4); 
            acc3 = _mm_max_pd(acc3, _mm_min_pd(c, _mm_min_pd(a, b)));

            a = _mm_loadu_pd(xx + k + 6);
            b = _mm_loadu_pd(yy + k + 6);
            c = _mm_loadu_pd(mask + k + 6); 
            acc4 = _mm_max_pd(acc4, _mm_min_pd(c, _mm_min_pd(a, b)));
        }

        // compute the remaining.
        max = (count - k) >> 1;
        switch (max) {
            case 3:  
                a = _mm_loadu_pd(xx + k + 4);
                b = _mm_loadu_pd(yy + k + 4);
                c = _mm_loadu_pd(mask + k + 4); 
                acc3 = _mm_max_pd(acc3, _mm_min_pd(c, _mm_min_pd(a, b)));
            case 2:
                a = _mm_loadu_pd(xx + k + 2);
                b = _mm_loadu_pd(yy + k + 2);
                c = _mm_loadu_pd(mask + k + 2); 
                acc2 = _mm_max_pd(acc2, _mm_min_pd(c, _mm_min_pd(a, b)));
            case 1:
                a = _mm_loadu_pd(xx + k);
                b = _mm_loadu_pd(yy + k);
                c = _mm_loadu_pd(mask + k); 
                acc1 = _mm_max_pd(acc1, _mm_min_pd(c, _mm_min_pd(a, b)));
            default: break;
        }
        k += (max << 1);

    } else {
        // compute the bulk
        for (; k < max; k += 8) {
            a = _mm_loadu_pd(xx + k);
            b = _mm_loadu_pd(yy + k);
            acc1 = _mm_max_pd(acc1, _mm_min_pd(a, b));

            a = _mm_loadu_pd(xx + k + 2);
            b = _mm_loadu_pd(yy + k + 2);
            acc2 = _mm_max_pd(acc2, _mm_min_pd(a, b));

            a = _mm_loadu_pd(xx + k + 4);
            b = _mm_loadu_pd(yy + k + 4);
            acc3 = _mm_max_pd(acc3, _mm_min_pd(a, b));

            a = _mm_loadu_pd(xx + k + 6);
            b = _mm_loadu_pd(yy + k + 6);
            acc4 = _mm_max_pd(acc4, _mm_min_pd(a, b));
        }

        // compute the remaining.
        max = (count - k) >> 1;
        switch (max) {
            case 3:  
                a = _mm_loadu_pd(xx + k + 4);
                b = _mm_loadu_pd(yy + k + 4);
                acc3 = _mm_max_pd(acc3, _mm_min_pd(a, b));
            case 2:
                a = _mm_loadu_pd(xx + k + 2);
                b = _mm_loadu_pd(yy + k + 2);
                acc2 = _mm_max_pd(acc2, _mm_min_pd(a, b));
            case 1:
                a = _mm_loadu_pd(xx + k);
                b = _mm_loadu_pd(yy + k);
                acc1 = _mm_max_pd(acc1, _mm_min_pd(a, b));
            default: break;
        }
        k += (max << 1);
    }
    // handle accumulators, extract data
    acc4 = _mm_max_pd(acc4, acc3);
    acc2 = _mm_max_pd(acc2, acc1);
    acc1 = _mm_max_pd(acc2, acc4);
    _mm_storeu_pd(val, acc1);

    // last 1 if there.
    if ((count & 1) > 0) {
        if (mask) {
            val[0] = std::max(val[0], std::min(mask[count - 1], std::min(xx[count - 1], yy[count - 1])));
        } else {
            val[0] = std::max(val[0], std::min(xx[count - 1], yy[count - 1]));
        }
    }

    return std::max(val[0], val[1]);

#else
    return std::numeric_limits<double>::lowest();

#endif

}   


inline double max_of_pairmin_avx(double const * xx, double const * yy, size_t const & count, double const * mask = nullptr) {
#ifdef __AVX__
    double val[4];

    __m256d a, b, c;
    __m256d acc1 = _mm256_set1_pd(std::numeric_limits<double>::lowest());
    __m256d acc2 = acc1;
    __m256d acc3 = acc1;
    __m256d acc4 = acc1;

    // compute the bulk
    size_t max = count & 0xFFFFFFFFFFFFFFF0;
    size_t k = 0;

    if (mask) {
        for (; k < max; k += 16) {
            a = _mm256_loadu_pd(xx + k);
            b = _mm256_loadu_pd(yy + k);
            c = _mm256_loadu_pd(mask + k); 
            acc1 = _mm256_max_pd(acc1, _mm256_min_pd(c, _mm256_min_pd(a, b)));

            a = _mm256_loadu_pd(xx + k + 4);
            b = _mm256_loadu_pd(yy + k + 4);
            c = _mm256_loadu_pd(mask + k + 4); 
            acc2 = _mm256_max_pd(acc2, _mm256_min_pd(c, _mm256_min_pd(a, b)));

            a = _mm256_loadu_pd(xx + k + 8);
            b = _mm256_loadu_pd(yy + k + 8);
            c = _mm256_loadu_pd(mask + k + 8); 
            acc3 = _mm256_max_pd(acc3, _mm256_min_pd(c, _mm256_min_pd(a, b)));

            a = _mm256_loadu_pd(xx + k + 12);
            b = _mm256_loadu_pd(yy + k + 12);
            c = _mm256_loadu_pd(mask + k + 12); 
            acc4 = _mm256_max_pd(acc4, _mm256_min_pd(c, _mm256_min_pd(a, b)));
        }

        // compute the remaining.
        max = (count - k) >> 2;
        switch (max) {
            case 3:  
                a = _mm256_loadu_pd(xx + k + 8);
                b = _mm256_loadu_pd(yy + k + 8);
                c = _mm256_loadu_pd(mask + k + 8); 
                acc3 = _mm256_max_pd(acc3, _mm256_min_pd(c, _mm256_min_pd(a, b)));
            case 2:
                a = _mm256_loadu_pd(xx + k + 4);
                b = _mm256_loadu_pd(yy + k + 4);
                c = _mm256_loadu_pd(mask + k + 4); 
                acc2 = _mm256_max_pd(acc2, _mm256_min_pd(c, _mm256_min_pd(a, b)));
            case 1:
                a = _mm256_loadu_pd(xx + k);
                b = _mm256_loadu_pd(yy + k);
                c = _mm256_loadu_pd(mask + k); 
                acc1 = _mm256_max_pd(acc1, _mm256_min_pd(c, _mm256_min_pd(a, b)));
            default: break;
        }
        k += (max << 2);

    } else {
        for (; k < max; k += 16) {
            a = _mm256_loadu_pd(xx + k);
            b = _mm256_loadu_pd(yy + k);
            acc1 = _mm256_max_pd(acc1, _mm256_min_pd(a, b));

            a = _mm256_loadu_pd(xx + k + 4);
            b = _mm256_loadu_pd(yy + k + 4);
            acc2 = _mm256_max_pd(acc2, _mm256_min_pd(a, b));

            a = _mm256_loadu_pd(xx + k + 8);
            b = _mm256_loadu_pd(yy + k + 8);
            acc3 = _mm256_max_pd(acc3, _mm256_min_pd(a, b));

            a = _mm256_loadu_pd(xx + k + 12);
            b = _mm256_loadu_pd(yy + k + 12);
            acc4 = _mm256_max_pd(acc4, _mm256_min_pd(a, b));
        }

        // compute the remaining.
        max = (count - k) >> 2;
        switch (max) {
            case 3:  
                a = _mm256_loadu_pd(xx + k + 8);
                b = _mm256_loadu_pd(yy + k + 8);
                acc3 = _mm256_max_pd(acc3, _mm256_min_pd(a, b));
            case 2:
                a = _mm256_loadu_pd(xx + k + 4);
                b = _mm256_loadu_pd(yy + k + 4);
                acc2 = _mm256_max_pd(acc2, _mm256_min_pd(a, b));
            case 1:
                a = _mm256_loadu_pd(xx + k);
                b = _mm256_loadu_pd(yy + k);
                acc1 = _mm256_max_pd(acc1, _mm256_min_pd(a, b));
            default: break;
        }
        k += (max << 2);
    }
    // handle accumulators, extract values
    acc4 = _mm256_max_pd(acc4, acc3);
    acc2 = _mm256_max_pd(acc2, acc1);
    acc1 = _mm256_max_pd(acc2, acc4);
    _mm256_storeu_pd(val, acc1);
 
    // last ones (up to 3) if there.
    if (mask) {
        for (size_t i = 0; k < count; ++i, ++k) {
            val[i] = std::max(val[i], std::min(mask[k], std::min(xx[k], yy[k])));
        }
    } else {
        for (size_t i = 0; k < count; ++i, ++k) {
            val[i] = std::max(val[i], std::min(xx[k], yy[k]));
        }
    }

    val[0] = std::max(val[0], val[1]);
    val[2] = std::max(val[2], val[3]);

    return std::max(val[0], val[2]);

#else
    return std::numeric_limits<double>::lowest();

#endif

}   

inline double max_of_pairmin_avx512(double const * xx, double const * yy, size_t const & count, double const * mask = nullptr) {

#ifdef __AVX512F__  
    double val[8];
    // has FMA, use it.

    __m512d a, b, c;
    __m512d acc1 = _mm512_set1_pd(std::numeric_limits<double>::lowest());
    __m512d acc2 = acc1;
    __m512d acc3 = acc1;
    __m512d acc4 = acc1;

    // compute the bulk
    size_t max = count & 0xFFFFFFFFFFFFFFE0;
    size_t k = 0;

    if (mask) {

        for (; k < max; k += 32) {
            a = _mm512_loadu_pd(xx + k);
            b = _mm512_loadu_pd(yy + k);
            c = _mm512_loadu_pd(mask + k);
            acc1 = _mm512_max_pd(acc1, _mm512_min_pd(c, _mm512_min_pd(a, b)));

            a = _mm512_loadu_pd(xx + k + 8);
            b = _mm512_loadu_pd(yy + k + 8);
            c = _mm512_loadu_pd(mask + k + 8);
            acc2 = _mm512_max_pd(acc2, _mm512_min_pd(c, _mm512_min_pd(a, b)));

            a = _mm512_loadu_pd(xx + k + 16);
            b = _mm512_loadu_pd(yy + k + 16);
            c = _mm512_loadu_pd(mask + k + 16);
            acc3 = _mm512_max_pd(acc3, _mm512_min_pd(c, _mm512_min_pd(a, b)));

            a = _mm512_loadu_pd(xx + k + 24);
            b = _mm512_loadu_pd(yy + k + 24);
            c = _mm512_loadu_pd(mask + k + 24);
            acc4 = _mm512_max_pd(acc4, _mm512_min_pd(c, _mm512_min_pd(a, b)));
        }

        // compute the remaining.
        max = (count - k) >> 3;
        switch (max) {
            case 3:  
                a = _mm512_loadu_pd(xx + k + 16);
                b = _mm512_loadu_pd(yy + k + 16);
                c = _mm512_loadu_pd(mask + k + 16);
                acc3 = _mm512_max_pd(acc3, _mm512_min_pd(c, _mm512_min_pd(a, b)));
            case 2:
                a = _mm512_loadu_pd(xx + k + 8);
                b = _mm512_loadu_pd(yy + k + 8);
                c = _mm512_loadu_pd(mask + k + 8);
                acc2 = _mm512_max_pd(acc2, _mm512_min_pd(c, _mm512_min_pd(a, b)));
            case 1:
                a = _mm512_loadu_pd(xx + k);
                b = _mm512_loadu_pd(yy + k);
                c = _mm512_loadu_pd(mask + k);
                acc1 = _mm512_max_pd(acc1, _mm512_min_pd(c, _mm512_min_pd(a, b)));
            default: break;
        }
        k += (max << 3);


    } else {

        for (; k < max; k += 32) {
            a = _mm512_loadu_pd(xx + k);
            b = _mm512_loadu_pd(yy + k);
            acc1 = _mm512_max_pd(acc1, _mm512_min_pd(a, b));

            a = _mm512_loadu_pd(xx + k + 8);
            b = _mm512_loadu_pd(yy + k + 8);
            acc2 = _mm512_max_pd(acc2, _mm512_min_pd(a, b));

            a = _mm512_loadu_pd(xx + k + 16);
            b = _mm512_loadu_pd(yy + k + 16);
            acc3 = _mm512_max_pd(acc3, _mm512_min_pd(a, b));

            a = _mm512_loadu_pd(xx + k + 24);
            b = _mm512_loadu_pd(yy + k + 24);
            acc4 = _mm512_max_pd(acc4, _mm512_min_pd(a, b));
        }

        // compute the remaining.
        max = (count - k) >> 3;
        switch (max) {
            case 3:  
                a = _mm512_loadu_pd(xx + k + 16);
                b = _mm512_loadu_pd(yy + k + 16);
                acc3 = _mm512_max_pd(acc3, _mm512_min_pd(a, b));
            case 2:
                a = _mm512_loadu_pd(xx + k + 8);
                b = _mm512_loadu_pd(yy + k + 8);
                acc2 = _mm512_max_pd(acc2, _mm512_min_pd(a, b));
            case 1:
                a = _mm512_loadu_pd(xx + k);
                b = _mm512_loadu_pd(yy + k);
                acc1 = _mm512_max_pd(acc1, _mm512_min_pd(a, b));
            default: break;
        }
        k += (max << 3);
    }
    // handle accumulators and extract
    acc4 = _mm512_max_pd(acc4, acc3);
    acc2 = _mm512_max_pd(acc2, acc1);
    acc1 = _mm512_max_pd(acc2, acc4);
    _mm512_storeu_pd(val, acc1);
 
    // last ones (up to 7) if there.
    if (mask) {
        for (size_t i = 0; k < count; ++i, ++k) {
            val[i] = std::max(val[i], std::min(mask[k], std::min(xx[k], yy[k])));
        }
    } else {
        for (size_t i = 0; k < count; ++i, ++k) {
            val[i] = std::max(val[i], std::min(xx[k], yy[k]));
        }
    }
    val[0] = std::max(val[0], val[1]);
    val[2] = std::max(val[2], val[3]);
    val[4] = std::max(val[4], val[5]);
    val[6] = std::max(val[6], val[7]);

    val[0] = std::max(val[0], val[2]);
    val[4] = std::max(val[4], val[6]);

    return std::max(val[0], val[4]);
#else
    return std::numeric_limits<double>::lowest();
#endif

}   


inline double max_of_pairmin(double const * xx, double const * yy, size_t const & count,
    double const * mask = nullptr) {
#if defined(__AVX512F__)
        return max_of_pairmin_avx512(xx, yy, count, mask);
#elif defined(__AVX__)
        return max_of_pairmin_avx(xx, yy, count, mask);
#elif defined(__SSE2__)
        return max_of_pairmin_sse(xx, yy, count, mask);
#else
        return max_of_pairmin_scalar(xx, yy, count, mask);
#endif

}

inline float max_of_pairmin(float const * xx, float const * yy, size_t const & count,
    float const * mask = nullptr) {
        return max_of_pairmin_scalar(xx, yy, count, mask);
}

// TODO.
template<typename T>
inline T max_of_pairmin(T const * xx, T const * yy, size_t const & count,
    std::set<size_t> const & excludes, T const * mask = nullptr) {

    T out = std::numeric_limits<T>::lowest();
    // get the first and second stops.
    size_t first = 0;
    size_t last, mask_first;

    for (size_t ex : excludes) {
        last = std::min(count, ex);
        mask_first = mask ? first : 0;
        if (last > first) {   // greater than 0
            out = std::max(out, max_of_pairmin(xx + first, yy + first, last - first, mask + mask_first));
        }
        first = last + 1;
    }
        
    last = count;
    mask_first = mask ? first : 0;
    if (last > first) {   // greater than 0
        out = std::max(out, max_of_pairmin(xx + first, yy + first, last - first, mask + mask_first));
    }

    return out;

}   


template<typename T>
inline T max_of_pairmin(T const * xx, T const * yy, size_t const & count,
    size_t const & exclude_x, size_t const & exclude_y, T const * mask = nullptr) {

    T out = std::numeric_limits<T>::lowest();
    // get the first and second stops.
    size_t first, last, mask_first;

    first = 0;
    last = std::min(count, std::min(exclude_x, exclude_y));
    mask_first = mask ? first : 0;
    if (last > first) {   // greater than 0
        out = std::max(out, max_of_pairmin(xx + first, yy + first, last - first, mask + mask_first));
    }
    
    first = last + 1;
    last = std::min(count, std::max(exclude_x, exclude_y));
    mask_first = mask ? first : 0;
    if (last > first) {   // greater than 0
        out = std::max(out, max_of_pairmin(xx + first, yy + first, last - first, mask + mask_first));
    }

    first = last + 1;
    last = count;
    mask_first = mask ? first : 0;
    if (last > first) {   // greater than 0
        out = std::max(out, max_of_pairmin(xx + first, yy + first, last - first, mask + mask_first));
    }

    return out;

}   


}}
