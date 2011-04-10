// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

/* This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License version 2
 * as published by the Free Software Foundation.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
 *
 * Authors: Caner Candan <caner@candan.fr>, http://caner.candan.fr
 */

#ifndef _linear_algebra_PowerMethod_h
#define _linear_algebra_PowerMethod_h

#include <cstdlib> // abs

#include <core_library/ConstBF.h>
#include <core_library/Continue.h>

#include "MultiplyMatVec.h"
#include "Dot.h"
#include "Scal.h"
#include "Norm.h"

namespace linear_algebra
{
    /**
       Base class for the power method. This a high level class implementing the power method function in keeping all operators used generic. Indeed you have to define the type of each operators used. Inherits from the const binary functor.
    */
    template < typename MatrixT, typename VectorT >
    class PowerMethod : public core_library::ConstBF< const MatrixT&, const VectorT&, typename MatrixT::AtomType >
    {
    public:
	typedef typename MatrixT::AtomType AtomType;

	//! main ctor
	PowerMethod( MultiplyMatVec< MatrixT, VectorT >& multiply, Dot< VectorT >& dot, Scal< VectorT >& scal, Norm< VectorT >& norm, core_library::Continue< AtomType >& continuator ) : _multiply(multiply), _dot(dot), _scal(scal), _norm(norm), _continuator( continuator ) {}

	//! main function
	virtual AtomType operator()( const MatrixT& A, const VectorT& x ) const
	{
	    AtomType lambda = 1.0;
	    AtomType old_lambda = 0.0;
	    VectorT y(x);

	    do
		{
		    old_lambda = lambda;
		    multiply(A,y,y);
		    lambda = dot(y,y);
		    scal(y,1/lambda);
		}
	    while ( _continuator( ::abs(old_lambda - lambda) ) );

	    return lambda;
	}

    private:
	MultiplyMatVec< MatrixT, VectorT >& _multiply;
	Dot< VectorT >& _dot;
	Scal< VectorT >& _scal;
	Norm< VectorT >& _norm;

	core_library::Continue< AtomType >& _continuator;
    };
}

#endif // !_linear_algebra_PowerMethod_h
