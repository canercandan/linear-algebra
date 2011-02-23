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

#ifndef _linear_algebra_detail_device_cublas_MatrixELL_h
#define _linear_algebra_detail_device_cublas_MatrixELL_h

#include <cusp/ell_matrix.h>

#include <linear_algebra/Matrix.h>

#include "Matrix.h"

namespace linear_algebra
{
    namespace detail
    {
	namespace device
	{
	    namespace cusp
	    {
		template < typename Atom >
		class MatrixELL : public Matrix< Atom, ::cusp::ell_matrix >
		{
		public:
		    MatrixELL() {}
		    MatrixELL(size_t n, size_t m, size_t nnz, size_t nnz_per_row, size_t align = 32) { this->resize(n, m, nnz, nnz_per_row, align); }

		    std::string className() const { return "MatrixELL"; }
		};
	    }
	}
    }
}

#endif // !_linear_algebra_detail_device_cublas_MatrixELL_h
