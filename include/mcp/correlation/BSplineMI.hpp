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
 * Author(s): Sriram Chockalingam, Tony C. Pan
 */

#pragma once

#include <vector>
#include "splash/kernel/kernel_base.hpp"
#include "splash/utils/memory.hpp"
#include "splash/kernel/dotproduct.hpp"


#if defined(USE_SIMD)
#include <omp.h>
#endif

// #define iLN_2 1.4426950408889634073599246810019L

namespace mcp { namespace kernel {

template<typename KT>
std::vector<KT> buildKnotVector(const int numBins, const int splineOrder) {
	std::vector<KT> v(numBins + splineOrder, 0.0);
	int nInternalPoints = numBins - splineOrder;
    
	int i;
    double norm_factor = static_cast<double>(1) / static_cast<double>(nInternalPoints + 1);

#if defined(__INTEL_COMPILER)
#pragma vector aligned
#endif
#if defined(USE_SIMD)
#pragma omp simd
#endif
	for (i = 0; i < splineOrder; ++i) {
		v[i] = static_cast<KT>(0);
	}
#if defined(__INTEL_COMPILER)
#pragma vector aligned
#endif
#if defined(USE_SIMD)
#pragma omp simd
#endif
	for (i = splineOrder; i < splineOrder + nInternalPoints; ++i) {
		v[i] = static_cast<KT>(i - splineOrder + 1) * norm_factor;
	}
#if defined(__INTEL_COMPILER)
#pragma vector aligned
#endif
#if defined(USE_SIMD)
#pragma omp simd
#endif
	for (i = splineOrder + nInternalPoints; i < 2*splineOrder + nInternalPoints; ++i) {
		v[i] = static_cast<KT>(1);
	}
	return v;
}


// for sample scaled score.  
template <typename IT, typename OT = IT, bool SampleStats = true>
class MinMaxScale : public splash::kernel::transform<IT, OT, splash::kernel::DEGREE::VECTOR> {
    public:
        using InputType = IT;
        using OutputType = OT;
        
        inline void operator()(IT const * __restrict__ in_vec, 
            size_t const & count,
            OT * __restrict__ out_vec) const {
            
            // compute minimum and maximum
            OT minX = count > 0 ? static_cast<OT>(in_vec[0]) : 0;
            OT maxX = count > 0 ? static_cast<OT>(in_vec[0]) : 0;
            OT x;
#if defined(__INTEL_COMPILER)
#pragma vector aligned
#endif
#if defined(USE_SIMD)
#pragma omp simd reduction(min:minX) reduction(max:maxX)
#endif
            for (size_t j = 1; j < count; ++j) {
                x = static_cast<OT>(in_vec[j]);
                minX = std::min(minX, x);
                maxX = std::max(maxX, x);
            }

            // compute the diff
            OT scale = (maxX > minX) ? 1.0 / (maxX - minX) : 1.0;
            OT offset = (maxX > minX) ? minX : 0.0;
            // normalize the data
#if defined(__INTEL_COMPILER)
#pragma vector aligned
#endif
#if defined(USE_SIMD)
#pragma omp simd
#endif
            for (size_t j = 0; j < count; ++j) {
                x = static_cast<OT>(in_vec[j]);
                out_vec[j] = scale * (x - offset);
            }
        }
};


template <typename IT, typename OT = IT>
class BSplineWeightsKernel : public splash::kernel::transform<IT, OT, splash::kernel::DEGREE::VECTOR> {

        int numBins;
        int splineOrder;
        int numSamples;
        std::vector<OT> knotVector;
        double norm_factor; 

        // Follows Daub et al, which contains mistakes;
        // corrections based on spline descriptions on MathWorld pages 
        OT basisFunction(const int i, const int p, OT const t) const {
            OT d1, n1, d2, n2, e1, e2;
            if (p == 1) {
                if ((t >= knotVector[i] && t < knotVector[i+1] && 
                    knotVector[i] < knotVector[i+1]) ||
                    (fabs(t - knotVector[i+1]) < 1e-10 && (i+1 == numBins))) {
                    return static_cast<OT>(1);
                }
                return static_cast<OT>(0);
            }
            
            d1 = knotVector[i+p-1] - knotVector[i];
            n1 = t - knotVector[i];
            d2 = knotVector[i+p] - knotVector[i+1];
            n2 = knotVector[i+p] - t;

            if (d1 < 1e-10 && d2 < 1e-10) {
                return static_cast<OT>(0);
            } else if (d1 < 1e-10) {
                e1 = static_cast<OT>(0);
                e2 = n2/d2*basisFunction(i+1, p-1, t);
            } else if (d2 < 1e-10) {
                e2 = static_cast<OT>(0);
                e1 = n1/d1*basisFunction(i, p-1, t);
            } else {
                e1 = n1/d1*basisFunction(i, p-1, t);
                e2 = n2/d2*basisFunction(i+1, p-1, t);
            }    
        
            // sometimes, this value is < 0 (only just; rounding error); truncate 
            if (e1 + e2 < 0) {
                return static_cast<OT>(0);
            }
            return(e1 + e2);
        }

        inline std::vector<OT> hist1d(OT const * __restrict__ weight_vec) const {
            std::vector<OT> hist(numBins, 0);
            
            for (int curBin = 0; curBin < numBins; curBin++) {
                OT ex = 0.0;
                const auto binBegin = curBin * numSamples;
#if defined(__INTEL_COMPILER)
#pragma vector aligned
#endif
#if defined(USE_SIMD)
#pragma omp simd reduction(+:ex)
#endif
                for (int curSample = 0; curSample < numSamples; curSample++) {
                    // ex = ex + ((weight_vec[binBegin + curSample]) / static_cast<OT>(numSamples));
                    ex += (weight_vec[binBegin + curSample]) * norm_factor;
                }
                hist[curBin] = ex;
            }
            return hist;
        }

        inline OT entropy1d(OT const * __restrict__ weight_vec) const {
            OT H = 0.0;
            std::vector<OT> hist = hist1d(weight_vec);
#if defined(__INTEL_COMPILER)
#pragma vector aligned
#endif
#if defined(USE_SIMD)
#pragma omp simd reduction(+:H)
#endif
            for (int curBin = 0; curBin < numBins; curBin++) {
                if (hist[curBin] > 0) {
                    H += hist[curBin] * log2(hist[curBin]);
                }
            }
            return 0.0 - H;
        }

        OT entropy1d_2(OT const * __restrict__ weight_vec) const {
            OT H = 0.0;
            int binBegin = 0;

            for (int curBin = 0; curBin < numBins; curBin++) {
                OT ex = 0.0;
#if defined(__INTEL_COMPILER)
#pragma vector aligned
#endif
#if defined(USE_SIMD)
#pragma omp simd reduction(+:ex)
#endif
                for (int curSample = 0; curSample < numSamples; curSample++) {
                    // ex = ex + ((weight_vec[binBegin + curSample]) / static_cast<OT>(numSamples));
                    ex += (weight_vec[binBegin]) * norm_factor;
                    ++binBegin;
                }
                if (ex > 0.0) H += ex * log2(ex);
            }
            return 0.0 - H;
        }

    public:

		BSplineWeightsKernel(int const & _bins, int const & _splineOrder,
                             int const & _numSamples) : 
             numBins(_bins), splineOrder(_splineOrder), numSamples(_numSamples), norm_factor(static_cast<double>(1) / static_cast<double>(_numSamples)) {
                 knotVector = buildKnotVector<OT>(numBins, splineOrder);
             }

        BSplineWeightsKernel() {}

        void copy_parameters(const BSplineWeightsKernel<IT,OT>& other){
            numBins = other.numBins;
            splineOrder = other.splineOrder;
            numSamples = other.numSamples;
            knotVector = other.knotVector;
            norm_factor = other.norm_factor; 
        }
        
		~BSplineWeightsKernel() {}

        size_t get_num_bins() const { return numBins; }

        inline void operator()(IT const * __restrict__ in_vec, size_t const & count,
            OT * __restrict__ out_vec) const {
            int num_weights = numSamples * numBins;    
#if defined(__INTEL_COMPILER)
#pragma vector aligned
#endif
#if defined(USE_SIMD)
#pragma omp simd
#endif
            for (int i = 0; i < num_weights; i++) {
                int curBin = i / numSamples;
                int curSample = i - (curBin * numSamples);
                OT x = static_cast<OT>(in_vec[curSample]);
                out_vec[i] = basisFunction(curBin, splineOrder, x);
            }
            // for (int curSample = 0; curSample < numSamples; curSample++) {
            //     for (int curBin = 0; curBin < numBins; curBin++) {
            //     OT x = static_cast<OT>(in_vec[curSample]);
            //     out_vec[curBin * numSamples + curSample] = 
            //             basisFunction(curBin, splineOrder, x);
            //    // mexPrintf("%d|%f(%f)\t", curBin, 
            //    //   weights[curBin * numSamples + curSample],z[curSample]); 
            //     }
            // }
            out_vec[num_weights] = entropy1d_2(out_vec);
        }
};

template<typename IT, typename OT = IT>
class BSplineEntropyKernel : public splash::kernel::inner_product<IT, OT, splash::kernel::DEGREE::VECTOR> {

    public:
        int numBins;
        int numSamples;

    protected:
        using FT = OT; // splash::utils::widened<OT>;
        FT norm_factor;
 
    public:        
        OT entropy2d_simd(IT const * wx, IT const * wy) const {
            int curBinX, curBinY;
            OT H = 0.0;
            int idx, idy;

            for (curBinX = 0; curBinX < numBins; ++curBinX) {

                for (curBinY = 0; curBinY < numBins; ++curBinY) {
                    idx = curBinX * numSamples;
                    idy = curBinY * numSamples;

                    FT hx;
    #if defined(__AVX512F__)
                    hx = splash::kernel::dotp_avx512(wx + idx, wy + idy, numSamples);
    #elif defined(__AVX__)
                    hx = splash::kernel::dotp_avx(wx + idx, wy + idy, numSamples);
    #elif defined(__SSE2__)
                    hx = splash::kernel::dotp_sse(wx + idx, wy + idy, numSamples);
    #else
                    hx = splash::kernel::dotp_scalar(wx + idx, wy + idy, numSamples);
    #endif
                    hx *= norm_factor;
                    if (hx > 0.0) H += hx * log2(hx);
                }
            }
            
            return 0.0 - H;
        }

		BSplineEntropyKernel(int const & _bins, 
                         int const & _numSamples) : 
             numBins(_bins), numSamples(_numSamples), norm_factor(static_cast<FT>(1) / static_cast<FT>(_numSamples)) {}

        BSplineEntropyKernel() {}

        void copy_parameters(const BSplineEntropyKernel<IT,OT>& other){
            numBins = other.numBins;
            numSamples = other.numSamples;
            norm_factor = other.norm_factor;
        }

		virtual ~BSplineEntropyKernel() {}

		inline OT operator()(IT const * first, IT const * second, size_t const & count) const  {
            assert((static_cast<int>(count) == numSamples * numBins + 1) && "Weight Vector not proper dimensions.");
            // std::cout<< "test" << std::endl;
            return entropy2d_simd(first, second);
        }
};


template<typename IT, typename OT = IT>
class BSplineMIKernel : public mcp::kernel::BSplineEntropyKernel<IT, OT> {

    using FT = OT; // splash::utils::widened<OT>;
 
    inline std::vector<OT> hist2d(IT const * wx, IT const * wy) const {
        int curSample;
        int curBinX, curBinY;
        std::vector<OT> output(this->numBins * this->numBins, 0.0);
        OT * out = output.data();
        IT const * x;
        IT const * y;
            
        for (curBinX = 0; curBinX < this->numBins; ++curBinX) {
            x = wx + curBinX * this->numSamples;
            for (curBinY = 0; curBinY < this->numBins; ++curBinY) {
                y = wy + curBinY * this->numSamples;

                FT hx = static_cast<FT>(0.0);
#if defined(__INTEL_COMPILER)
#pragma vector aligned
#endif
#if defined(USE_SIMD)
#pragma omp simd reduction(+:hx)
#endif
                for (curSample = 0; curSample < this->numSamples; curSample++) {
                    hx += this->norm_factor * static_cast<FT>(x[curSample] * y[curSample]); 
                        // static_cast<OT>(wy[binBeginY + curSample]) / static_cast<OT>(numSamples);
                }
                *out = hx; // was += hx, but this should be same.
                ++out;
            }
        }
            
        /*
            for (curBinX = 0; curBinX < numBinsX; curBinX++) {
            for (curBinY = 0; curBinY < numBinsY; curBinY++) {
            mexPrintf("%f\t", hist[curBinX * numBinsY + curBinY]);
            }
            mexPrintf("\n");
            }
        */
        return output;
    }


    inline OT entropy2d(IT const * wx, IT const * wy) const {
        OT H = 0;

        std::vector<OT> hist = hist2d(wx, wy);
#if defined(__INTEL_COMPILER)
#pragma vector aligned
#endif
#if defined(USE_SIMD)
#pragma omp simd reduction(+:H)
#endif            
        for (size_t i = 0; i < hist.size(); ++i) {
            OT incr = hist[i];
            if (incr > 0) {
                H += incr * log2(incr);
            }
        }
        return 0.0 - H;
    }

    // old method, 1k x 1k:  16.89s 
    // combined, 1k x 1k: 18.74s
    OT entropy2d_2(IT const * wx, IT const * wy) const {
        int curSample;
        int curBinX, curBinY;
        OT H = 0.0;
        int idx, idy;

        for (curBinX = 0; curBinX < this->numBins; ++curBinX) {
            

            for (curBinY = 0; curBinY < this->numBins; ++curBinY) {
                idy = curBinY * this->numSamples;
                idx = curBinX * this->numSamples;

                FT hx = static_cast<FT>(0.0);
// #if defined(__INTEL_COMPILER)
// #pragma vector aligned
// #endif
// #if defined(USE_SIMD)
// #pragma omp simd reduction(+:hx)
// #endif
                for (curSample = 0; curSample < this->numSamples; curSample++) {
                    hx += this->norm_factor * static_cast<FT>(wx[idx] * wy[idy]); 
                        // static_cast<OT>(wy[binBeginY + curSample]) / static_cast<OT>(numSamples);
                    ++idx;
                    ++idy;
                }
                if (hx > 0.0) H += hx * log2(hx);
            }
        }
           
        return 0.0 - H;
    }

    OT entropy2d_omp(IT const * wx, IT const * wy) const {
        OT H = 0.0;
        FT hx;
        int idx, idy;

        size_t count = this->numBins * this->numSamples;
        IT * wx2 = splash::utils::aalloc<IT>(count << 1);
        memcpy(wx2, wx, count * sizeof(IT));
        memcpy(wx2 + count, wx, count * sizeof(IT));
        

        for (int offset = 0; offset < this->numBins; ++offset) {
            idx = offset * this->numSamples;
            idy = 0;
            for (int bin = 0; bin < this->numBins; ++bin) {
                hx = 0.0;

#if defined(__INTEL_COMPILER)
#pragma vector aligned
#endif
#if defined(USE_SIMD)
#pragma omp simd reduction(+:hx)
#endif
                for (int s = 0; s < this->numSamples; ++s) {
                    hx += this->norm_factor * static_cast<FT>(wx2[idx] * wy[idy]); 
                    ++idx;
                    ++idy;
                }
                if (hx > 0.0) H += hx * log2(hx);
            }
        }
           
        return 0.0 - H;
    }


	public:
		BSplineMIKernel(int const & _bins, 
                         int const & _numSamples) : 
             mcp::kernel::BSplineEntropyKernel<IT, OT>(_bins, _numSamples) {}

        BSplineMIKernel() {}

        void copy_parameters(const BSplineMIKernel<IT,OT>& other){
            this->BSplineEntropyKernel<IT, OT>::copy_parameters(other);
            // numBins = other.numBins;
            // numSamples = other.numSamples;
            // norm_factor = other.norm_factor;
        }

		virtual ~BSplineMIKernel() {}

        // computed as H(X) + H(Y) - H(X, Y)
		inline OT operator()(IT const * first, IT const * second, size_t const & count) const  {
            assert((static_cast<int>(count) == this->numSamples * this->numBins + 1) && "Weight Vector not proper dimensions.");
            // std::cout<< "test" << std::endl;
            OT e2d = this->entropy2d_simd(first, second);
            int e1idx = this->numSamples * this->numBins;
            return static_cast<OT>(first[e1idx]) + 
                static_cast<OT>(second[e1idx]) - e2d;
        }
};

}}
