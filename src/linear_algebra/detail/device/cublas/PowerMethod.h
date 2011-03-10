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
		    typedef linear_algebra::PowerMethod< Matrix< Atom >, Vector< Atom > > Parent;

		public:
		    PowerMethod() : Parent( _default_multiply, _default_dot, _default_scal, _default_norm, _default_continuator ) {}

		    PowerMethod( core_library::Continue< Atom >& continuator ) : Parent( _default_multiply, _default_dot, _default_scal, _default_norm, continuator ) {}

		    PowerMethod( MultiplyMatVec< Atom >& multiply, Dot< Atom >& dot, Scal< Atom >& scal, Norm< Atom >& norm ) : Parent( multiply, dot, scal, norm, _default_continuator ) {}

		    PowerMethod( MultiplyMatVec< Atom >& multiply, Dot< Atom >& dot, Scal< Atom >& scal, Norm< Atom >& norm, core_library::Continue< Atom >& continuator ) : Parent( multiply, dot, scal, norm, continuator ) {}

		private:
		    MultiplyMatVec< Atom > _default_multiply;
		    Dot< Atom > _default_dot;
		    Scal< Atom > _default_scal;
		    Norm< Atom > _default_norm;

		    core_library::DummyContinue< Atom > _default_continuator;
		};
	    }
	}
    }
}

#endif // !_linear_algebra_detail_device_cublas_PowerMethod_h
