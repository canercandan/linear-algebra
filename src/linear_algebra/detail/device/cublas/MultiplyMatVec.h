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

#ifndef _linear_algebra_detail_device_cublas_MultiplyMatVec_h
#define _linear_algebra_detail_device_cublas_MultiplyMatVec_h

#include <stdexcept>

#include <cublas.h>

#include <linear_algebra/MultiplyMatVec.h>

#include "Vector.h"
#include "Matrix.h"
#include "MultiplyMatVecOp.h"
#include "Gemv.h"

namespace linear_algebra
{
    namespace detail
    {
	namespace device
	{
	    namespace cublas
	    {
		template < typename Atom >
		class MultiplyMatVec : public linear_algebra::MultiplyMatVec< Matrix< Atom >, Vector< Atom > >
		{
		public:
		    MultiplyMatVec() : _operation( Gemv< Atom >() ) {}
		    MultiplyMatVec( MultiplyMatVecOperation& operation ) : _operation( operation ) {}

		    void operator()( const Matrix< Atom >& A, const Vector< Atom >& x, Vector< Atom >& y )
		    {
			int n = A.rows();
			if ( y.size() < n ) { y.resize( n ); }

			_operation(A, x, y);

			cublasStatus stat = cublasGetError();
			if ( stat != CUBLAS_STATUS_SUCCESS )
			    {
				throw std::runtime_error("matvec failed");
			    }
		    }

		private:
		    MultiplyMatVecOp< Atom >& _operation;
		};
	    }
	}
    }
}

#endif // !_linear_algebra_detail_device_cublas_MultiplyMatVec_h
