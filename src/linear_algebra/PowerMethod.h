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

#include "Gemv.h"
#include "Dot.h"
#include "Scal.h"

template < typename Atom >
class PowerMethod
{
public:
    Scalar< Atom > operator()( Matrix< Atom >& A, Vector< Atom >& w0, Atom epsilon, int nbItMax )
    {
	int nb = 0;
	Scalar< Atom > lambda = 0;
	Scalar< Atom > lambda_old = 0;
	while ( abs(old_lambda - lambda) > epsilon &&
		nb < nbItMax )
	    {
		gemv(A,v,w);
		lambda_old = lambda;
		lambda = dot(w);
		lambda = sqrt(lambda);
		scal(w,v,lambda);
		nb++;
	    }
    }

private:
    const Gemv& gemv;
    const Dot& dot;
    const Scal& scal;
};

#endif // !_linear_algebra_PowerMethod_h
