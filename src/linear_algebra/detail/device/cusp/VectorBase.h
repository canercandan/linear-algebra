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

#ifndef _linear_algebra_detail_device_cublas_VectorBase_h
#define _linear_algebra_detail_device_cublas_VectorBase_h

#include <cusp/memory.h>
#include <cusp/print.h>

#include <linear_algebra/Vector.h>

namespace linear_algebra
{
    namespace detail
    {
	namespace device
	{
	    namespace cusp
	    {
		template < typename Format >
		class VectorBase : public linear_algebra::Vector< typename Format::AtomType >, public Format
		{
		public:
		    typedef typename Format::AtomType AtomType;
		    typedef Format FormatType;

		    using Format::resize;
		    using Format::operator=;

		    std::string className() const { return "VectorBase"; }

		    virtual void printOn(std::ostream& _os) const
		    {
			::cusp::print_matrix( *this );
		    }
		};
	    }
	}
    }
}

#endif // !_linear_algebra_detail_device_cublas_VectorBase_h
