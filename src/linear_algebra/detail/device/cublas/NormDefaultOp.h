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

#ifndef _linear_algebra_detail_device_cublas_NormDefaultOp_h
#define _linear_algebra_detail_device_cublas_NormDefaultOp_h

#include <cublas.h>

#include <linear_algebra/detail/device/cuda/cuda>

#include "NormOp.h"

namespace linear_algebra
{
    namespace detail
    {
	namespace device
	{
	    namespace cublas
	    {
		template < typename Atom >
		class NormDefaultOp : public NormOp< Atom >
		{
		public:
		    Atom operator()( const Vector< Atom >& array ) const;
		};

		template <>
		float NormDefaultOp< float >::operator()( const Vector< float >& array ) const { return cublasSnrm2( array.size(), array, 1 ); }

		template <>
		double NormDefaultOp< double >::operator()( const Vector< double >& array ) const { return cublasDnrm2( array.size(), array, 1 ); }

		template <>
		cuda::SingleComplex NormDefaultOp< cuda::SingleComplex >::operator()( const Vector< cuda::SingleComplex >& array ) const { return cublasScnrm2( array.size(), array, 1 ); }

		template <>
		cuda::DoubleComplex NormDefaultOp< cuda::DoubleComplex >::operator()( const Vector< cuda::DoubleComplex >& array ) const { return cublasDznrm2( array.size(), array, 1 ); }
	    }
	}
    }
}

#endif // !_linear_algebra_detail_device_cublas_NormDefaultOp_h
