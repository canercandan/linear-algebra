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

#include <linear_algebra/linear_algebra>
#include <linear_algebra/detail/device/cusp/cusp>

using namespace linear_algebra::detail::device::cusp;

typedef float Atom;
//typedef double Atom;

int main(void)
{
    const int N = 4;

    Vector< Atom > x(N, 1);

    NormDefaultOp< Vector< Atom > > op;
    Norm< Vector< Atom > > _norm(op);
    core_library::logger << "norm result: " << _norm( x ) << std::endl;

    //core_library::logger << "norm result: " << norm( x ) << std::endl;

    return 0;
}
