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

#include <linear_algebra/detail/device/cublas/cublas>
#include <linear_algebra/detail/device/cuda/cuda>

using namespace linear_algebra::detail::device::cublas;
using namespace linear_algebra::detail::device::cuda;

typedef SingleComplex T;
//typedef double T;

int main(void)
{
    T sc1(12.0);
    T sc2(12.0);

    T scr = sc1 + sc2;

    core_library::logger << scr << std::endl;

    const int N = 10;
    const int M = 10;

    Matrix<T> A(M, N, scr);
    Vector<T> x(N, scr);
    // Vector<T> y;

    MultiplyMatVec<T> multiply;

    // multiply( A, x, y );

    // core_library::logger << "size: " << y.size() << std::endl;
    // core_library::logger << y << std::endl;

    return 0;
}
