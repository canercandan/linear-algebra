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
//typedef double Atom;

int main(void)
{
    const int N = 4;

    Vector<T> x(N, 20);

    // Scal< Vector< Atom > > scal;
    // scal(x,12.0f);

    scal(x,T(12.0f, 0.0f));

    core_library::logger << "scal result: " << x << std::endl;

    return 0;
}
