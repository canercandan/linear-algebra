// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
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

#ifndef _linear_algebra_ConstTF_h
#define _linear_algebra_ConstTF_h

#include "FunctorBase.h"

namespace linear_algebra
{

    /**
       Const Basic Tertiary Functor. Derive from this class when defining
       any tertiary function. First template argument is result_type, second
       is first_argument_type, third is second_argument_type.
       Argument and result types can be any type including void for
       result_type
    **/
    template <class A1, class A2, class A3, class R>
    class ConstTF : public FunctorBase
    {
    public :
        /// virtual dtor here so there is no need to define it in derived classes
	virtual ~ConstTF() {}

	//typedef R result_type;
	//typedef A1 first_argument_type;
	//typedef A2 second_argument_type;

	/// The pure virtual function that needs to be implemented by the subclass
	virtual R operator()(const A1&, const A2&, const A3&) const = 0;

	/// tag to identify a procedure in compile time function selection @see functor_category
	static FunctorBase::tertiary_function_tag functor_category()
	{
	    return FunctorBase::tertiary_function_tag();
	}
    };

    /**
       Overloaded function that can help in the compile time detection
       of the type of functor we are dealing with
       @see Counter, make_counter
    */
    template<class R, class A1, class A2, class A3>
    FunctorBase::tertiary_function_tag functor_category(const ConstTF<A1, A2, A3, R>&)
    {
	return FunctorBase::tertiary_function_tag();
    }

}

#endif // !_linear_algebra_ConstTF_h
