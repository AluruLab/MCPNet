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

#include <functional>
#include "splash/ds/aligned_matrix.hpp"
#include "splash/ds/aligned_vector.hpp"
#include "splash/kernel/kernel_base.hpp"

// filter a matrix or vector by a fixed value.


namespace mcp { namespace kernel {


// filter by pvalue or by correlation only
template <typename IT, bool negate = false>
class mask : public splash::kernel::binary_op<IT, bool, IT, splash::kernel::DEGREE::VECTOR> {
    protected: 
        IT default_val;

    public:
        mask(IT const & _default = 0) : default_val(_default) {}
        virtual ~mask() {}

        void copy_parameters(mask const & other) {
            default_val = other.default_val;
        }

        inline virtual void operator()(IT const * in_vec, bool const * mas, size_t const & count, IT * out_vec) const {
            // negate == false:  negate ^ mas = mas
            // negate == true:   negate ^ mas = !mas
            for (size_t i = 0; i < count; ++i) {
                out_vec[i] = (negate ^ mas[i]) ? in_vec[i] : this->default_val;
                this->processed += (out_vec[i] != in_vec[i]);
            }
        }
};


// filter by single value, e.g. pvalue or by correlation only
template <typename IT>
class inverted_threshold : public splash::kernel::transform<IT, IT, splash::kernel::DEGREE::VECTOR> {
    protected: 
        IT default_val;
        IT min_thresh;
        IT max_thresh;

    public:
        inverted_threshold(IT const & _min_thresh = std::numeric_limits<IT>::lowest(), 
            IT const & _max_thresh = std::numeric_limits<IT>::max(), 
            IT const & _default = 0) : default_val(_default), min_thresh(_min_thresh), max_thresh(_max_thresh) {}
        virtual ~inverted_threshold() {};

        void copy_parameters(inverted_threshold const & other) {
            default_val = other.default_val;
            min_thresh = other.min_thresh;
            max_thresh = other.max_thresh;
        }

        inline virtual void operator()(IT const * in_vector, 
            size_t const & count,
            IT * out_vector) const {
            for (size_t i = 0; i < count; ++i) {
                IT in = in_vector[i];
                out_vector[i] = ((in <= min_thresh) || (in >= max_thresh)) ? in : this->default_val;
                this->processed += (out_vector[i] != in);
            }
        }
};

template <typename IT>
class threshold : public splash::kernel::transform<IT, IT, splash::kernel::DEGREE::VECTOR> {
    protected: 
        IT default_val;
        IT min_thresh;
        IT max_thresh;

    public:
        threshold(IT const & _min_thresh = std::numeric_limits<IT>::lowest(), 
            IT const & _max_thresh = std::numeric_limits<IT>::max(), 
            IT const & _default = 0) : default_val(_default), min_thresh(_min_thresh), max_thresh(_max_thresh) {}
        virtual ~threshold() {};

        void copy_parameters(threshold const & other) {
            default_val = other.default_val;
            min_thresh = other.min_thresh;
            max_thresh = other.max_thresh;
        }

        inline virtual void operator()(IT const * in_vector, 
            size_t const & count,
            IT * out_vector) const {
            for (size_t i = 0; i < count; ++i) {
                IT in = in_vector[i];
                out_vector[i] = ((in < min_thresh) || (in > max_thresh)) ? this->default_val : in;
                this->processed += (out_vector[i] != in);
            }
        }
};


// filter by a predicate matrix.
template <typename IT, typename IT1>
class inverted_threshold2 : public splash::kernel::binary_op<IT, IT1, IT, splash::kernel::DEGREE::VECTOR>{
    protected: 
        IT default_val;
        IT min_thresh;
        IT max_thresh;

    public:
        inverted_threshold2(IT const & _min_thresh = std::numeric_limits<IT>::lowest(),
            IT const & _max_thresh  = std::numeric_limits<IT>::max(),
            IT const & _default = 0) : default_val(_default), 
                min_thresh(_min_thresh), max_thresh(_max_thresh) {}
        virtual ~inverted_threshold2() {};

        void copy_parameters(inverted_threshold2 const & other) {
            default_val = other.default_val;
            min_thresh = other.min_thresh;
            max_thresh = other.max_thresh;
        }

        inline virtual void operator()(IT const * in, IT1 const * aux1, size_t const & count, IT * out) const {
            for (size_t i = 0; i < count; ++i) {
                IT x = in[i];
                IT1 aux = aux1[i];
                out[i] = ((aux <= this->min_thresh) || (aux >= this->max_thresh)) ? x : this->default_val;
                this->processed += (out[i] != x);
            }
        }
};


// filter by a predicate matrix.
template <typename IT, typename IT1>
class threshold2 : public splash::kernel::binary_op<IT, IT1, IT, splash::kernel::DEGREE::VECTOR>{
    protected: 
        IT default_val;
        IT min_thresh;
        IT max_thresh;

    public:
        threshold2(IT const & _min_thresh = std::numeric_limits<IT>::lowest(),
            IT const & _max_thresh  = std::numeric_limits<IT>::max(),
            IT const & _default = 0) : default_val(_default), 
                min_thresh(_min_thresh), max_thresh(_max_thresh) {}
        virtual ~threshold2() {};

        void copy_parameters(threshold2 const & other) {
            default_val = other.default_val;
            min_thresh = other.min_thresh;
            max_thresh = other.max_thresh;
        }

        inline virtual void operator()(IT const * in, IT1 const * aux1, size_t const & count, IT * out) const {
            for (size_t i = 0; i < count; ++i) {
                IT x = in[i];
                IT1 aux = aux1[i];
                out[i] = ((aux < this->min_thresh) || (aux > this->max_thresh)) ? this->default_val : x;
                this->processed += (out[i] != x);
            }
        }
};


}}
