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

#ifndef _linear_algebra_detail_device_cuda_Complex_h
#define _linear_algebra_detail_device_cuda_Complex_h

#include <vector_types.h>  // required for float2, double2

#include <linear_algebra/Complex.h>

namespace linear_algebra
{
    // template <>
    // M_HOSTDEVICE Complex< SingleComplex >::operator cuComplex() const { return _value; }

    // template <>
    // M_HOSTDEVICE Complex< DoubleComplex >::operator cuDoubleComplex() const { return _value; }

    namespace detail
    {
	namespace device
	{
	    namespace cuda
	    {
		//! here's the common complex types you can use
		typedef linear_algebra::Complex< float2, float > SingleComplex;
		typedef linear_algebra::Complex< double2, double > DoubleComplex;
		//typedef Complex< doublesingle2, doublesingle > DoubleSingleComplex;
	    }
	}
    }
}

#endif // !_linear_algebra_detail_device_cuda_Complex_h
