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

#ifndef _linear_algebra_detail_device_cublas_Vector_h
#define _linear_algebra_detail_device_cublas_Vector_h

#include <linear_algebra/Vector.h>

#include "Array.h"

namespace linear_algebra
{
    namespace detail
    {
	namespace device
	{
	    namespace cublas
	    {
		template < typename Atom >
		class Vector : public Array< Atom >, virtual public linear_algebra::Vector< Atom >
		{
		public:
		    Vector() {}
		    Vector(int n) : Array< Atom >(n) {}
		    Vector(int n, Atom value) : Array< Atom >(n, value) {}

		    std::string className() const { return "Vector"; }
		};
	    }
	}
    }
}

#endif // !_linear_algebra_detail_device_cublas_Vector_h
