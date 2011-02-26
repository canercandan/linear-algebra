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

#ifndef _linear_algebra_detail_device_cusp_MultiplyMatVec_h
#define _linear_algebra_detail_device_cusp_MultiplyMatVec_h

#include <stdexcept>

#include <cusp/multiply.h>

#include <linear_algebra/MultiplyMatVec.h>

#include "VectorBase.h"
#include "MatrixBase.h"
#include "MultiplyMatVecOp.h"
#include "MultiplyMatVecDefaultOp.h"

namespace linear_algebra
{
    namespace detail
    {
	namespace device
	{
	    namespace cusp
	    {
		template < typename MatrixT, typename VectorT >
		class MultiplyMatVec : public linear_algebra::MultiplyMatVec< MatrixBase< typename MatrixT::FormatType >, VectorBase< typename VectorT::FormatType > >
		{
		public:
		    MultiplyMatVec() : _operation( _default_operation ) {}
		    MultiplyMatVec( MultiplyMatVecOp< MatrixT, VectorT >& operation ) : _operation( operation ) {}

		    void operator()( const MatrixBase< typename MatrixT::FormatType >& A, const VectorBase< typename VectorT::FormatType >& x, VectorBase< typename VectorT::FormatType >& y )
		    {
			int n = A.num_rows;
			if ( y.size() < n ) { y.resize( n ); }

			_operation(A, x, y);
		    }

		private:
		    MultiplyMatVecDefaultOp< MatrixT, VectorT > _default_operation;
		    MultiplyMatVecOp< MatrixT, VectorT >& _operation;
		};

		template < typename MatrixT, typename VectorT >
		void multiply( const MatrixT& A, const VectorT& x, VectorT& y ) { MultiplyMatVec< MatrixT, VectorT >()(A,x,y); }
	    }
	}
    }
}

#endif // !_linear_algebra_detail_device_cusp_MultiplyMatVec_h
