// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

/* This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License version 2
 * as published by the Free Software Foundation.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
 *
 * Authors: Caner Candan <caner@candan.fr>, http://caner.candan.fr
 */

#ifndef _linear_algebra_Matrix_h
#define _linear_algebra_Matrix_h

#include "core_library/Object.h"
#include "core_library/Printable.h"

#include "Array.h"

using namespace core_library;

namespace linear_algebra
{
    /**
       Base class for matrix structure, inherits from the Array base class.
    */
    template < typename Atom >
    class Matrix : virtual public Array< Atom >
    {
    public:
	//virtual Matrix& operator=( const Matrix< Atom >& v ) = 0;

	// virtual int rows() const = 0;
	// virtual int cols() const = 0;
    };
}

#endif // !_linear_algebra_Matrix_h
