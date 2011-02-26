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

#ifndef _linear_algebra_detail_device_cusp_PowerMethod_h
#define _linear_algebra_detail_device_cusp_PowerMethod_h

#include <linear_algebra/PowerMethod.h>

namespace linear_algebra
{
    namespace detail
    {
	namespace device
	{
	    namespace cusp
	    {
		template < typename MatrixT, typename VectorT >
		class PowerMethod : public linear_algebra::PowerMethod< MatrixBase< typename MatrixT::FormatType >, VectorBase< typename VectorT::FormatType > >
		{
		public:
		    typedef typename MatrixT::AtomType AtomType;

		    PowerMethod( AtomType epsilon, int nbItMax ) : _epsilon(epsilon), _nbItMax(nbItMax), _multiply(_default_multiply), _dot(_default_dot), _scal(_default_scal), _norm(_default_norm) {}

		    PowerMethod( AtomType epsilon, int nbItMax, MultiplyMatVec< MatrixT, VectorT >& multiply, Dot< VectorT > dot, Scal< VectorT > scal, Norm< VectorT > norm ) : _epsilon(epsilon), _nbItMax(nbItMax), _multiply(multiply), _dot(dot), _scal(scal), _norm(norm) {}

		    AtomType operator()( const MatrixBase< typename MatrixT::FormatType >& A, const VectorBase< typename VectorT::FormatType >& x ) const
		    {
			// TODO
		    }

		private:
		    MultiplyMatVec< MatrixT, VectorT > _default_multiply;
		    Dot< VectorT > _default_dot;
		    Scal< VectorT > _default_scal;
		    Norm< VectorT > _default_norm;

		    AtomType _epsilon;
		    int _nbItMax;
		    MultiplyMatVec< MatrixT, VectorT >& _multiply;
		    Dot< VectorT >& _dot;
		    Scal< VectorT >& _scal;
		    Norm< VectorT >& _norm;
		};

		template < typename MatrixT, typename VectorT >
		typename MatrixT::AtomType powermethod( const MatrixT& A, const VectorT& x ) { return PowerMethod< MatrixT, VectorT >()(A,x); }
	    }
	}
    }
}

#endif // !_linear_algebra_detail_device_cusp_PowerMethod_h
