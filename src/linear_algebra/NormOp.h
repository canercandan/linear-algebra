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

#ifndef _linear_algebra_NormOp_h
#define _linear_algebra_NormOp_h

#include "core_library/ConstUF.h"

namespace linear_algebra
{
    /**
       Base class for norm operation, inherits from the const unary fonctor. This is used by Norm classes in order to get genericity with operator.
    */
    template < typename VectorT >
    class NormOp : public core_library::ConstUF< VectorT, typename VectorT::AtomType > {};
}

#endif // !_linear_algebra_NormOp_h
