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

#ifndef _linear_algebra_detail_device_cublas_Scal_h
#define _linear_algebra_detail_device_cublas_Scal_h

#include <cublas.h>

#include <linear_algebra/Scal.h>

#include "Vector.h"

namespace linear_algebra
{
    namespace detail
    {
	namespace device
	{
	    namespace cublas
	    {
		template < typename Atom >
		class Scal : public linear_algebra::Scal< Vector< Atom > >
		{
		public:
		    void operator()( Vector< Atom >& array, Atom alpha );
		};

		template <>
		void Scal< float >::operator()( Vector< float >& array, float alpha ) { return cublasSscal( array.size(), alpha, array, 1 ); }

		template <>
		void Scal< double >::operator()( Vector< double >& array, double alpha ) { return cublasDscal( array.size(), alpha, array, 1 ); }

		template < typename Atom >
		void scal( Vector< Atom >& array, Atom alpha ) { Scal< Atom >()(array,alpha); }
	    }
	}
    }
}

#endif // !_linear_algebra_detail_device_cublas_Scal_h
