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

#include <cusp/memory.h>
#include <cusp/array1d.h>

#include <linear_algebra/Vector.h>

#include "VectorBase.h"

namespace linear_algebra
{
    namespace detail
    {
	namespace device
	{
	    namespace cusp
	    {
		template < typename Atom >
		class VectorWrapperArray1d : public ::cusp::array1d< Atom, ::cusp::device_memory >
		{
		public:
		    typedef Atom AtomType;
		};

		template < typename Atom >
		class Vector : public VectorBase< VectorWrapperArray1d< Atom > >
		{
		public:
		    Vector() {}
		    Vector(size_t n) { this->resize(n); }
		    Vector(size_t n, const Atom& value) { this->resize(n, value); }

		    std::string className() const { return "Vector"; }
		};
	    }
	}
    }
}

#endif // !_linear_algebra_detail_device_cublas_Vector_h
