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

#ifndef _linear_algebra_detail_device_cublas_Dot_h
#define _linear_algebra_detail_device_cublas_Dot_h

#include <cublas.h>

#include <linear_algebra/Dot.h>
#include <linear_algebra/detail/device/cuda/cuda>

#include "Vector.h"

namespace linear_algebra
{
    namespace detail
    {
	namespace device
	{
	    namespace cublas
	    {
		/**
		   Dot for cublas
		*/
		template < typename Atom >
		class Dot : public linear_algebra::Dot< Vector< Atom > >
		{
		public:
		    //! main function
		    Atom operator()( const Vector< Atom >& x, const Vector< Atom >& y ) const;
		};

		/**
		   main function specialized for single precision
		*/
		template <>
		float Dot< float >::operator()( const Vector< float >& x, const Vector< float >& y ) const { return cublasSdot( x.size(), x, 1, y, 1 ); }

		/**
		   main function specialized for double precision
		*/
		template <>
		double Dot< double >::operator()( const Vector< double >& x, const Vector< double >& y ) const { return cublasDdot( x.size(), x, 1, y, 1 ); }

		/**
		   main function specialized for single precision complex number
		*/
		template <>
		cuda::SingleComplex Dot< cuda::SingleComplex >::operator()( const Vector< cuda::SingleComplex >& x, const Vector< cuda::SingleComplex >& y ) const { return cublasCdotu( x.size(), x, 1, y, 1 ); }

		/**
		   main function specialized for double precision complex number
		*/
		template <>
		cuda::DoubleComplex Dot< cuda::DoubleComplex >::operator()( const Vector< cuda::DoubleComplex >& x, const Vector< cuda::DoubleComplex >& y ) const { return cublasZdotu( x.size(), x, 1, y, 1 ); }

		/**
		   the function version of the dot operator
		*/
		template < typename Atom >
		Atom dot( const Vector< Atom >& x, const Vector< Atom >& y ) { return Dot< Atom >()(x,y); }
	    }
	}
    }
}

#endif // !_linear_algebra_detail_device_cublas_Dot_h
