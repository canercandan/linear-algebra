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

#ifndef _linear_algebra_Complex_h
#define _linear_algebra_Complex_h

#include <core_library/Printable.h>

/**
   This class has been largely inspired by the cudacomplex
   class developed by Christian Buchner. Thanks to him.
*/

/**
   In order to easily defined in which context the functions
   are going to be used, there are some defines below indicating
   the host and device case. Otherwise we're just defining the
   functions static.
*/
#undef HOST
#undef DEVICE
#undef HOSTDEVICE
#undef M_HOST
#undef M_HOSTDEVICE

#ifdef __CUDACC__
# define HOST		__host__
# define DEVICE		__device__
# define HOSTDEVICE	__host__ __device__
# define M_HOST		__host__
# define M_HOSTDEVICE	__host__ __device__
#else
# define HOST		static inline
# define DEVICE		static inline
# define HOSTDEVICE	static inline
# define M_HOST		inline
# define M_HOSTDEVICE	inline
#endif

/**
   Some routines to make compatible CUDA compiler with others.
*/
#ifdef __CUDACC__
# define ALIGN(x)  __align__(x)
#else
# if defined(_MSC_VER) && (_MSC_VER >= 1300) // Visual C++ .NET and later
#  define ALIGN(x) __declspec(align(x))
# else
#  if defined(__GNUC__) // GCC
#   define ALIGN(x)  __attribute__ ((aligned (x)))
#  else // all other compilers
#   define ALIGN(x)
#  endif
# endif
#endif

namespace linear_algebra
{
    /**
       Generic complex class,
       Free to use complex class in single or double precision.
    */
    template < typename ComplexType, typename AtomType >
    class Complex : public core_library::Printable
    {
    public:
	typedef ComplexType complex_type;
	typedef AtomType atom_type;
	typedef Complex< ComplexType, AtomType > struct_type;

	//! Main constructor
	Complex( const AtomType real = 0.0, const AtomType imag = 0.0 )
	{
	    _value.x = real;
	    _value.y = imag;
	}

	//! copy constructor from ComplexType
	Complex( const ComplexType cplx )
	{
	    _value.x = cplx.x;
	    _value.y = cplx.y;
	}

	//! Assignment from scalar to complex
	Complex< ComplexType, AtomType >& operator=( const AtomType& scalar )
	{
	    _value.x = scalar;
	    _value.y = 0;
	    return *this;
	}

	//! Assignment from pair to complex
	struct_type* operator=( const std::pair< AtomType, AtomType >& p )
	{
	    _value.x = p.first;
	    _value.y = p.second;
	    return *this;
	}

	//! get the real
	M_HOSTDEVICE AtomType& real() { return _value.x; }

	//! get the imaginar
	M_HOSTDEVICE AtomType& imag() { return _value.y; }

	// get the real in const
	M_HOSTDEVICE AtomType& real() const { return _value.x; }

	//! get the imaginar in const
	M_HOSTDEVICE AtomType& imag() const { return _value.y; }

	//! creation and addition with a complex number
	M_HOSTDEVICE struct_type operator+( const struct_type& complex ) const
	{
	    return struct_type( _value.x + complex._value.x, _value.y + complex._value.y );
	}

	//! addition with a complex number
	M_HOSTDEVICE struct_type operator+=( const struct_type& complex ) const
	{
	    _value.x += complex._value.x;
	    _value.y += complex._value.y;
	    return *this;
	}

	M_HOSTDEVICE struct_type operator+( const AtomType& real ) const
	{
	    return struct_type( _value.x + real, _value.y );
	}

	//! subtract with a complex number in creating a new one
	M_HOSTDEVICE struct_type operator-( const struct_type& complex ) const
	{
	    return struct_type( _value.x - complex._value.x, _value.y - complex._value.y );
	}

	//! negate complex in creating a new one
	M_HOSTDEVICE struct_type operator-() const
	{
	    return struct_type( -_value.x, -_value.y );
	}

	//! subtract scalar from complex in creating a new one
	M_HOSTDEVICE struct_type operator-( const AtomType& real ) const
	{
	    return struct_type( _value.x - real, _value.y );
	}

	//! multiply two complex numbers
	M_HOSTDEVICE struct_type operator*( const struct_type& complex ) const
	{
	    return struct_type( _value.x * complex._value.x - _value.y * complex._value.y,
				_value.y * complex._value.x + _value.x * complex._value.y );
	}

	//! multiply complex with real
	M_HOSTDEVICE struct_type operator*( const AtomType& real ) const
	{
	    return struct_type( _value.x * real, _value.y * real );
	}

	//! divide two complex numbers
	M_HOSTDEVICE struct_type operator/( const struct_type& complex ) const
	{
	    AtomType tmp = complex._value.x * complex._value.x + complex._value.y * complex._value.y;
	    return struct_type( _value.x * complex._value.x + _value.y * complex._value.y,
				_value.y * complex._value.x - _value.x * complex._value.y );
	}

	//! divide complex with real
	M_HOSTDEVICE struct_type operator/( const AtomType& real ) const
	{
	    return struct_type( _value.x / real, _value.y / real );
	}

	//! conjugate complex
	M_HOSTDEVICE struct_type operator~() const
	{
	    return struct_type( _value.x, -_value.y );
	}

	//! comparaison with real
	M_HOSTDEVICE bool operator==( const AtomType& real ) const
	{
	    return _value.x == real && _value.y == 0;
	}

	//! complex modulus and absolute
	M_HOSTDEVICE AtomType abs() const
	{
	    return sqrt( _value.x * _value.x + _value.y * _value.y );
	}

	//! complex phase angle
	M_HOSTDEVICE struct_type phase()
	{
	    return __atan2( _value.y, _value.x );
	}

	//! return constant number one
	static M_HOSTDEVICE struct_type one() { return struct_type(1); }

	//! return constant number zero
	static M_HOSTDEVICE struct_type zero() { return struct_type(); }

	//! return constant number I
	static M_HOSTDEVICE struct_type I() { return struct_type(0, 1); }

	//! return the primitive type
	M_HOSTDEVICE operator ComplexType() const { return _value; }

	virtual void printOn(std::ostream& os) const
	{
	    os << "C(" << _value.x << "," << _value.y << ")";
	}

    private:
	ComplexType _value;
    };

    //! a overload of abs for complex numbers
    template <typename ComplexType, typename AtomType>
    HOSTDEVICE AtomType abs( const Complex< ComplexType, AtomType >& z ) { return z.abs(); }

    //! a overload of sqrt for complex numbers
    template <typename ComplexType, typename AtomType>
    HOSTDEVICE Complex< ComplexType, AtomType > sqrt( const Complex< ComplexType, AtomType >& z )
    {
	AtomType x = z.real();
	AtomType y = z.imag();

	if (x == AtomType())
	    {
		AtomType t = sqrt( abs(y) / 2 );
		return Complex< ComplexType, AtomType >( t, y < AtomType() ? -t : t );
	    }
	else
	    {
		AtomType t = sqrt( 2 * ( abs(z) + abs(x) ) );
		AtomType u = t / 2;
		return x > AtomType()
		    ? Complex< ComplexType, AtomType >( u, y / t )
		    : Complex< ComplexType, AtomType >( abs(y) / t, y < AtomType() ? -u : u );
	    }
    }

    //! a overload of comparaison operator < for complex numbers
    template <typename ComplexType, typename AtomType>
    HOSTDEVICE bool operator<( const Complex< ComplexType, AtomType >& a,
			       const Complex< ComplexType, AtomType >& b )
    {
	return a.real() < b.real();
    }
}

#endif // !_linear_algebra_Complex_h
