// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// FunctorBase.h
// (c) Maarten Keijzer 2000
/*
    This library is free software; you can redistribute it and/or
    modify it under the terms of the GNU Lesser General Public
    License as published by the Free Software Foundation; either
    version 2 of the License, or (at your option) any later version.

    This library is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
    Lesser General Public License for more details.

    You should have received a copy of the GNU Lesser General Public
    License along with this library; if not, write to the Free Software
    Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA

    Contact: todos@geneura.ugr.es, http://geneura.ugr.es
             mak@dhi.dk
    CVS Info: $Date: 2004-12-01 09:22:48 $ $Header: /home/nojhan/dev/eodev/eodev_cvs/eo/src/Functor.h,v 1.7 2004-12-01 09:22:48 evomarc Exp $ $Author: evomarc $
 */
//-----------------------------------------------------------------------------

#ifndef _linear_algebra_FunctorBase_h
#define _linear_algebra_FunctorBase_h

#include <functional>

namespace linear_algebra
{

    /** @addtogroup Core
     * @{
     */

    /** Base class for functors to get a nice hierarchy diagram

	That's actually quite an understatement as it does quite a bit more than
	just that. By having all functors derive from the same base class, we can
	do some memory management that would otherwise be very hard.

	The memory management base class is called FunctorStore, and it supports
	a member add() to add a pointer to a functor. When the functorStore is
	destroyed, it will delete all those pointers. So beware: do not delete
	the functorStore before you are done with anything that might have been allocated.

	@see FunctorStore

    */
    class FunctorBase
    {
    public :
	/// virtual dtor here so there is no need to define it in derived classes
	virtual ~FunctorBase() {}

	/// tag to identify a procedure in compile time function selection @see functor_category
	struct procedure_tag {};
	/// tag to identify a unary function in compile time function selection @see functor_category
	struct unary_function_tag {};
	/// tag to identify a binary function in compile time function selection @see functor_category
	struct binary_function_tag {};
    };
    /** @example t-Functor.cpp
     */

    /** @} */

}

#endif // !_linear_algebra_FunctorBase_h
