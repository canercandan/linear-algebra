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

#ifndef _linear_algebra_MultiplyMatVec_h
#define _linear_algebra_MultiplyMatVec_h

#include "core_library/BO.h"

namespace linear_algebra
{
    /**
       Base class for the matrix-vector product, inherits from the binary operation class.
    */
    template < typename MatrixT, typename VectorT >
    class MultiplyMatVec : public core_library::BO< MatrixT, VectorT, VectorT > {};
}

#endif // !_linear_algebra_MultiplyMatVec_h
