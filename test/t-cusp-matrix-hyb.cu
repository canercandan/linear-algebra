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

#include <linear_algebra/detail/device/cusp/cusp>

using namespace linear_algebra::detail::device::cusp;
using namespace core_library;

typedef float Atom;

int main(void)
{
    // allocate storage for (4,3) matrix with 8 nonzeros
    //     ELL portion has 5 nonzeros and storage for 2 nonzeros per row
    //     COO portion has 3 nonzeros
    MatrixHYB< Atom > A(3,4,5,3,2);

    // Initialize A to represent the following matrix
    // [10  20  30  40]
    // [ 0  50   0   0]
    // [60   0  70  80]

    // A is split into ELL and COO parts as follows
    // [10  20  30  40]    [10  20   0   0]     [ 0   0  30  40]
    // [ 0  50   0   0]  = [ 0  50   0   0]  +  [ 0   0   0   0]
    // [60   0  70  80]    [60   0  70   0]     [ 0   0   0  80]

    // Initialize ELL part

    // X is used to fill unused entries in the ELL portion of the matrix
    const int X = MatrixELL< Atom >::invalid_index;

    // first row
    A.ell.column_indices(0,0) = 0; A.ell.values(0,0) = 10;
    A.ell.column_indices(0,1) = 1; A.ell.values(0,1) = 20;

    // second row
    A.ell.column_indices(1,0) = 1; A.ell.values(1,0) = 50;  // shifted to leftmost position
    A.ell.column_indices(1,1) = X; A.ell.values(1,1) =  0;  // padding

    // third row
    A.ell.column_indices(2,0) = 0; A.ell.values(2,0) = 60;
    A.ell.column_indices(2,1) = 2; A.ell.values(2,1) = 70;  // shifted to leftmost position

    // Initialize COO part
    A.coo.row_indices[0] = 0;  A.coo.column_indices[0] = 2;  A.coo.values[0] = 30;
    A.coo.row_indices[1] = 0;  A.coo.column_indices[1] = 3;  A.coo.values[1] = 40;
    A.coo.row_indices[2] = 2;  A.coo.column_indices[2] = 3;  A.coo.values[2] = 80;

    logger << A;

    return 0;
}
