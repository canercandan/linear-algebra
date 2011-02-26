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

#ifndef _linear_algebra_detail_device_cusp_Scal_h
#define _linear_algebra_detail_device_cusp_Scal_h

#include <stdexcept>

#include <cusp/blas.h>

#include <linear_algebra/Scal.h>

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
		class Scal : public linear_algebra::Scal< VectorBase< typename VectorT::FormatType > >
		{
		public:
		    void operator()( VectorBase< typename VectorT::FormatType >& array, typename VectorT::AtomType alpha ) { ::cusp::blas::scal(array, alpha); }
		};

		template < typename VectorT >
		void scal( VectorT& array, typename VectorT::AtomType alpha ) { Scal< VectorT >()(array,alpha); }
	    }
	}
    }
}

#endif // !_linear_algebra_detail_device_cusp_Scal_h
