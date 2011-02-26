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

#ifndef _linear_algebra_detail_device_cusp_Dot_h
#define _linear_algebra_detail_device_cusp_Dot_h

#include <stdexcept>

#include <cusp/blas.h>

#include <linear_algebra/Dot.h>

#include "VectorBase.h"

namespace linear_algebra
{
    namespace detail
    {
	namespace device
	{
	    namespace cusp
	    {
		template < typename VectorT >
		class Dot : public linear_algebra::Dot< VectorBase< typename VectorT::FormatType > >
		{
		public:
		    typename VectorT::AtomType operator()( const VectorBase< typename VectorT::FormatType >& x, const VectorBase< typename VectorT::FormatType >& y ) const { return ::cusp::blas::dot(x, y); }
		};

		template < typename VectorT >
		typename VectorT::AtomType dot( const VectorT& x, const VectorT& y ) { return Dot< VectorT >()(x,y); }
	    }
	}
    }
}

#endif // !_linear_algebra_detail_device_cusp_Dot_h
