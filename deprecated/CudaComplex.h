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

#ifndef _CudaComplex_h
#define _CudaComplex_h

/*
 * This class has been largely inspired by the cudacomplex
 * class developed by Christian Buchner. Thanks to him.
 */

#include <vector_types.h>  // required for float2 and double2

/*
 * In order to easily defined in which context the functions
 * are going to be used, there are some defines below indicating
 * the host and device case. Otherwise we're just defining the
 * functions static.
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
#else
# define HOST		static inline
# define DEVICE		static inline
# define HOSTDEVICE	static inline
# define M_HOST		inline
# define M_HOSTDEVICE	inline
#endif

/*
 * Some routines to make compatible CUDA compiler with others.
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

namespace LA
{

    /*
     * Cuda generic complex class,
     * Free to use complex class in single or double precision.
     */
    typename < typename ComplexType, typename AtomType >
    class CudaComplex
    {
    public:
	typedef ComplexType complex_type;
	typedef AtomType atom_type;
	typedef CudaComplex< ComplexType, AtomType > struct_type;

	//! Main constructor
	CudaComplex( const AtomType real = 0.0, const AtomType imag = 0.0 )
	{
	    _value.x = real;
	    _value.y = imag;
	}

	//! Assignment from scalar to complex
	CudaComplex< ComplexType, AtomType >& operator=( const AtomType& scalar )
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
	}

	//! get the real
	HOSTDEVICE AtomType& real() { return _value.x; }

	//! get the imaginar
	HOSTDEVICE AtomType& imag() { return _value.y; }

	// get the real in const
	HOSTDEVICE AtomType& real() const { return _value.x; }

	//! get the imaginar in const
	HOSTDEVICE AtomType& imag() const { return _value.y; }

	//! creation and addition with a complex number
	HOSTDEVICE struct_type operator+( const struct_type& complex ) const
	{
	    return struct_type( _value.x + complex.x, _value.y + complex.y );
	}

	//! addition with a complex number
	HOSTDEVICE struct_type operator+=( const struct_type& complex ) const
	{
	    _value.x += complex.x;
	    _value.y += complex.y;
	    return *this;
	}

	HOSTDEVICE struct_type operator+( const AtomType& real ) const
	{
	    return struct_type( _value.x + real, _value.y );
	}

	//! subtract with a complex number in creating a new one
	HOSTDEVICE struct_type operator-( const struct_type& complex ) const
	{
	    return struct_type( _value.x - complex.x, _value.y - complex.y );
	}

	//! negate complex in creating a new one
	HOSTDEVICE struct_type operator-() const
	{
	    return struct_type( -_value.x, -_value.y );
	}

	//! subtract scalar from complex in creating a new one
	HOSTDEVICE struct_type operator-( const AtomType& real ) const
	{
	    return struct_type( _value.x - real, _value.y );
	}

	//! multiply two complex numbers
	HOSTDEVICE struct_type operator*( const struct_type& complex ) const
	{
	    return struct_type( _value.x * complex.x - _value.y * complex.y,
				_value.y * complex.x + _value.x * complex.y );
	}

	//! multiply complex with real
	HOSTDEVICE struct_type operator*( const AtomType& real ) const
	{
	    return struct_type( _value.x * real, _value.y * b );
	}

	//! divide two complex numbers
	HOSTDEVICE struct_type operator/( const struct_type& complex ) const
	{
	    AtomType tmp = complex.x * complex.x + complex.y * complex.y;
	    return struct_type( _value.x * complex.x + _value.y * complex.y,
				_value.y * complex.x - _value.x * complex.y );
	}

	//! divide complex with real
	HOSTDEVICE struct_type operator/( const AtomType& real ) const
	{
	    return struct_type( _value.x / real, _value.y / b );
	}

	//! conjugate complex
	HOSTDEVICE struct_type operator~() const
	{
	    return struct_type( _value.x, -_value.y );
	}

	//! comparaison with real
	HOSTDEVICE bool operator==( const AtomType& real ) const
	{
	    return _value.x == real && _value.y == 0;
	}

	//! complex modulus and absolute
	HOSTDEVICE AtomType abs() const
	{
	    return sqrt( _value.x * _value.x + _value.y * _value.y );
	}

	//! complex phase angle
	HOSTDEVICE struct_type phase()
	{
	    return __atan2( _value.y, _value.x );
	}

	//! return constant number one
	static HOSTDEVICE struct_type one() { return struct_type(1); }

	//! return constant number zero
	static HOSTDEVICE struct_type zero() { return struct_type(); }

	//! return constant number I
	static HOSTDEVICE struct_type I() { return struct_type(0, 1); }

	//! return the primitive CUDA type
	operator ComplexType() const { return _value; }

    private:
	ComplexType _value;
    };

    //! a overload of abs for complex numbers
    template <typename ComplexType, typename AtomType>
    AtomType abs( const CudaComplex< ComplexType, AtomType >& z ) { return z.abs(); }

    //! a overload of sqrt for complex numbers
    template <typename ComplexType, typename AtomType>
    CudaComplex< ComplexType, AtomType > sqrt( const CudaComplex< ComplexType, AtomType > REF(z) )
    {
	AtomType x = z.real();
	AtomType y = z.imag();

	if (x == AtomType())
	    {
		AtomType t = sqrt( abs(y) / 2 );
		return CudaComplex< ComplexType, AtomType >( t, y < AtomType() ? -t : t );
	    }
	else
	    {
		AtomType t = sqrt( 2 * ( abs(z) + abs(x) ) );
		AtomType u = t / 2;
		return x > AtomType()
		    ? CudaComplex< ComplexType, AtomType >( u, y / t )
		    : CudaComplex< ComplexType, AtomType >( abs(y) / t, y < AtomType() ? -u : u );
	    }
    }

    //! a overload of comparaison operator < for complex numbers
    template <typename ComplexType, typename AtomType>
    bool operator<( const CudaComplex< ComplexType, AtomType >& a,
		    const CudaComplex< ComplexType, AtomType >& a )
    {
	return a.real() < b.real();
    }

    //! here's the common complex types you can use
    typedef CudaComplex< float2, float > CudaSingleComplex;
    typedef CudaComplex< double2, double2 > CudaDoubleComplex;
    typedef CudaComplex< doublesingle2, doublesingle > CudaDoubleSingleComplex;

}

#endif // !_CudaComplex_h
