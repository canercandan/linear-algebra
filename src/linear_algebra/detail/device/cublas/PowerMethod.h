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

#ifndef _linear_algebra_detail_device_cublas_PowerMethod_h
#define _linear_algebra_detail_device_cublas_PowerMethod_h

#include <core_library/Continue.h>

#include <linear_algebra/PowerMethod.h>

#include "Vector.h"
#include "Matrix.h"
#include "MultiplyMatVec.h"
#include "Dot.h"
#include "Scal.h"
#include "Norm.h"

namespace linear_algebra
{
    namespace detail
    {
	namespace device
	{
	    namespace cublas
	    {
		template < typename Atom >
		class PowerMethod : public linear_algebra::PowerMethod< Matrix< Atom >, Vector< Atom > >
		{
		public:
		    PowerMethod() : _multiply(_default_multiply), _dot(_default_dot), _scal(_default_scal), _norm(_default_norm), _continuator( _default_continuator ) {}

		    PowerMethod( core_library::Continue< Atom >& continuator ) : _multiply(_default_multiply), _dot(_default_dot), _scal(_default_scal), _norm(_default_norm), _continuator( continuator ) {}

		    PowerMethod( MultiplyMatVec< Atom >& multiply, Dot< Atom > dot, Scal< Atom > scal, Norm< Atom > norm ) : _multiply(multiply), _dot(dot), _scal(scal), _norm(norm), _continuator( _default_continuator ) {}

		    PowerMethod( MultiplyMatVec< Atom >& multiply, Dot< Atom > dot, Scal< Atom > scal, Norm< Atom > norm, core_library::Continue< Atom >& continuator ) : _multiply(multiply), _dot(dot), _scal(scal), _norm(norm), _continuator( continuator ) {}

		    Atom operator()( const Matrix< Atom >& A, const Vector< Atom >& x ) const
		    {
			Atom lambda = 0;
			Atom old_lambda = 0;

			do
			    {
				old_lambda = lambda;
				Vector< Atom > y;
				multiply(A,x,y);
				lambda = ::sqrt( dot(y,y) );
				scal(w,lambda);
			    }
			while ( _continuator( ::abs(old_lambda - lambda) ) );

			return lambda;
		    }

		private:
		    MultiplyMatVec< Atom > _default_multiply;
		    Dot< Atom > _default_dot;
		    Scal< Atom > _default_scal;
		    Norm< Atom > _default_norm;
		    core_library::DummyContinue< Atom > _default_continuator;

		    MultiplyMatVec< Atom >& _multiply;
		    Dot< Atom >& _dot;
		    Scal< Atom >& _scal;
		    Norm< Atom >& _norm;

		    core_library::Continue< Atom >& _continuator;
		};
	    }
	}
    }
}

#endif // !_linear_algebra_detail_device_cublas_PowerMethod_h
