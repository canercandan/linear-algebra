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
    // allocate storage for (4,3) matrix with 6 nonzeros in 3 diagonals
    MatrixDIA< Atom > A(4,3,6,3);

    // initialize diagonal offsets
    A.diagonal_offsets[0] = -2;
    A.diagonal_offsets[1] =  0;
    A.diagonal_offsets[2] =  1;

    // initialize diagonal values

    // first diagonal
    A.values(0,2) =  0;  // outside matrix
    A.values(1,2) =  0;  // outside matrix
    A.values(2,0) = 40;
    A.values(3,0) = 60;

    // second diagonal
    A.values(0,1) = 10;
    A.values(1,1) =  0;
    A.values(2,1) = 50;
    A.values(3,1) = 50;  // outside matrix

    // third diagonal
    A.values(0,2) = 20;
    A.values(1,2) = 30;
    A.values(2,2) =  0;  // outside matrix
    A.values(3,2) =  0;  // outside matrix

    // A now represents the following matrix
    //    [10 20  0]
    //    [ 0  0 30]
    //    [40  0 50]
    //    [ 0 60  0]

    logger << A;

    return 0;
}
