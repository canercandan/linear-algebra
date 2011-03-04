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

#ifndef _linear_algebra_detail_device_cublas_Gemv_h
#define _linear_algebra_detail_device_cublas_Gemv_h

#include <cublas.h>
#include <cuComplex.h>

#include <linear_algebra/detail/device/cuda/cuda>

#include "MultiplyMatVecOp.h"

namespace linear_algebra
{
    namespace detail
    {
	namespace device
	{
	    using cuda::SingleComplex;
	    using cuda::DoubleComplex;

	    namespace cublas
	    {
		template < typename Atom >
		class Gemv : public MultiplyMatVecOp< Atom >
		{
		public:
		    void operator()( const Matrix< Atom >&, const Vector< Atom >&, Vector< Atom >& );
		};

		template <>
		void Gemv< float >::operator()( const Matrix< float >& A, const Vector< float >& x, Vector< float >& y )
		{
		    int n = A.rows();
		    int m = A.cols();
		    cublasSgemv('n', m, n, 1, A, m, x, 1, 0, y, 1);
		}

		template <>
		void Gemv< double >::operator()( const Matrix< double >& A, const Vector< double >& x, Vector< double >& y )
		{
		    int n = A.rows();
		    int m = A.cols();
		    cublasDgemv('n', m, n, 1, A, m, x, 1, 0, y, 1);
		}

		template <>
		void Gemv< SingleComplex >::operator()( const Matrix< SingleComplex >& A, const Vector< SingleComplex >& x, Vector< SingleComplex >& y )
		{
		    int n = A.rows();
		    int m = A.cols();
		    cublasCgemv('n', m, n, SingleComplex(1), A, m, x, 1, SingleComplex(0), y, 1);
		}

		template <>
		void Gemv< DoubleComplex >::operator()( const Matrix< DoubleComplex >& A, const Vector< DoubleComplex >& x, Vector< DoubleComplex >& y )
		{
		    int n = A.rows();
		    int m = A.cols();
		    cublasZgemv('n', m, n, DoubleComplex(1), A, m, x, 1, DoubleComplex(0), y, 1);
		}
	    }
	}
    }
}

#endif // !_linear_algebra_detail_device_cublas_Gemv_h
