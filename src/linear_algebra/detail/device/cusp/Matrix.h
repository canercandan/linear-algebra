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

#ifndef _linear_algebra_detail_device_cublas_Matrix_h
#define _linear_algebra_detail_device_cublas_Matrix_h

#include <cusp/memory.h>
#include <cusp/array2d.h>

#include "MatrixBase.h"

namespace linear_algebra
{
    namespace detail
    {
	namespace device
	{
	    namespace cusp
	    {
		template < typename Atom >
		class MatrixWrapperArray2d : public ::cusp::array2d< Atom, ::cusp::host_memory >
		{
		public:
		    typedef Atom AtomType;
		};

		template < typename Atom >
		class Matrix : public MatrixBase< MatrixWrapperArray2d< Atom > >
		{
		public:
		    Matrix() {}
		    Matrix(size_t n, size_t m) { this->resize(n, m); }
		    Matrix(size_t n, size_t m, const Atom& value)
		    {
			this->resize(n, m);
			this->values.resize(n*m, value);
		    }

		    std::string className() const { return "Matrix"; }
		};
	    }
	}
    }
}

#endif // !_linear_algebra_detail_device_cublas_Matrix_h
