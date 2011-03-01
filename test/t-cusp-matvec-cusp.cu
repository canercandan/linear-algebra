// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

/* This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.

 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.

 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 * Authors:
 * Caner Candan <caner@candan.fr>, http://caner.candan.fr
 */

/// Examples are powered by cusp-library

#include <iostream>

#include <linear_algebra/detail/device/cusp/cusp>

#include <cusp/csr_matrix.h>

using namespace linear_algebra::detail::device::cusp;
using namespace core_library;

typedef float Atom;

int main(void)
{
    // allocate storage for (4,3) matrix with 4 nonzeros
    ::cusp::csr_matrix< int, Atom, ::cusp::device_memory > A(4,3,6);

    // initialize matrix entries on host
    A.row_offsets[0] = 0;  // first offset is always zero
    A.row_offsets[1] = 2;
    A.row_offsets[2] = 2;
    A.row_offsets[3] = 3;
    A.row_offsets[4] = 6; // last offset is always num_entries

    A.column_indices[0] = 0; A.values[0] = 10;
    A.column_indices[1] = 2; A.values[1] = 20;
    A.column_indices[2] = 2; A.values[2] = 30;
    A.column_indices[3] = 0; A.values[3] = 40;
    A.column_indices[4] = 1; A.values[4] = 50;
    A.column_indices[5] = 2; A.values[5] = 60;

    // A now represents the following matrix
    //    [10  0 20]
    //    [ 0  0  0]
    //    [ 0  0 30]
    //    [40 50 60]

    //logger << A;

    ::cusp::array1d< Atom, ::cusp::device_memory > x(3,1);
    ::cusp::array1d< Atom, ::cusp::device_memory > y(4);

    ::cusp::multiply(A,x,y);

    ::cusp::print_matrix(y);

    return 0;
}
