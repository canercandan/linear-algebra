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

#ifndef _linear_algebra_detail_device_cublas_Norm_h
#define _linear_algebra_detail_device_cublas_Norm_h

#include <cublas.h>

#include <linear_algebra/Norm.h>

#include "Vector.h"
#include "NormOp.h"
#include "NormDefaultOp.h"

namespace linear_algebra
{
    namespace detail
    {
	namespace device
	{
	    namespace cublas
	    {
		template < typename Atom >
		class Norm : public linear_algebra::Norm< Vector< Atom > >
		{
		public:
		    Norm() : _operation( _default_operation ) {}
		    Norm( NormOp< Atom >& operation ) : _operation( operation ) {}

		    Atom operator()( const Vector< Atom >& array ) const { return _operation( array ); }

		private:
		    NormDefaultOp< Atom > _default_operation;
		    NormOp< Atom >& _operation;
		};

		template < typename Atom >
		Atom norm( const Vector< Atom >& array ) { return Norm< Atom >()(array); }
	    }
	}
    }
}

#endif // !_linear_algebra_detail_device_cublas_Norm_h
