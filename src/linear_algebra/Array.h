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

#ifndef _linear_algebra_Array_h
#define _linear_algebra_Array_h

#include "core_library/Object.h"
#include "core_library/Printable.h"

using namespace core_library;

namespace linear_algebra
{
    /**
       Base class for vectors and matrixes, it requires to be printable.
    */
    template < typename Atom >
    class Array : public Object, public Printable
    {
    public:
	typedef Atom AtomType;

	//virtual Array& operator=( const Array< Atom >& v ) = 0;

	//virtual operator Atom*() const = 0;

	// virtual int size() const = 0;
	//virtual void resize(int size) = 0;
    };
}

#endif // !_linear_algebra_Array_h
