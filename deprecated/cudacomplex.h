/*
* Copyright (c) 2008-2009 Christian Buchner <Christian.Buchner@gmail.com>
* All rights reserved.
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions are met:
*        * Redistributions of source code must retain the above copyright
*          notice, this list of conditions and the following disclaimer.
*        * Redistributions in binary form must reproduce the above copyright
*          notice, this list of conditions and the following disclaimer in the
*          documentation and/or other materials provided with the distribution.
*
* THIS SOFTWARE IS PROVIDED BY Christian Buchner ''AS IS'' AND ANY 
* EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
* WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
* DISCLAIMED. IN NO EVENT SHALL Christian Buchner BE LIABLE FOR ANY
* DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
* (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
* LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
* ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
* (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
* SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#ifndef CUDACOMPLEX_H
#define CUDACOMPLEX_H

#include <vector_types.h>  // required for float2, double2

// Depending on whether we're running inside the CUDA compiler, define the __host_
// and __device__ intrinsics, otherwise just make the functions static to prevent
// linkage issues (duplicate symbols and such)
#ifdef __CUDACC__
#define HOST __host__
#define DEVICE __device__
#define HOSTDEVICE __host__ __device__
#define M_HOST __host__
#define M_HOSTDEVICE __host__ __device__
#else
#define HOST static inline
#define DEVICE static inline
#define HOSTDEVICE static inline
#define M_HOST inline      // note there is no static here
#define M_HOSTDEVICE inline // (static has a different meaning for class member functions)
#endif

// Struct alignment is handled differently between the CUDA compiler and other
// compilers (e.g. GCC, MS Visual C++ .NET)
#ifdef __CUDACC__
#define ALIGN(x)  __align__(x)
#else
#if defined(_MSC_VER) && (_MSC_VER >= 1300)
// Visual C++ .NET and later
#define ALIGN(x) __declspec(align(x))
#else
#if defined(__GNUC__)
// GCC
#define ALIGN(x)  __attribute__ ((aligned (x)))
#else
// all other compilers
#define ALIGN(x)
#endif
#endif
#endif

// Somehow in emulation mode the code won't compile Mac OS X 1.1 CUDA SDK when the
// operators below make use of references (compiler bug?). So instead we compile
// the code to pass everything through the stack. Slower, but works.
// I am not sure how the Linux CUDA SDK will behave, so currently when I detect
// Microsoft's Visual C++.NET I always allow it to use references.
#if !defined(__DEVICE_EMULATION__) || (defined(_MSC_VER) && (_MSC_VER >= 1300))
#define REF(x) &x
#define ARRAYREF(x,y) (&x)[y]
#define PTR(x) *x
#else
#define REF(x) x
#define ARRAYREF(x,y) x[y]
#define PTR(x) *x
#endif


/**
 * A templated complex number type for use with CUDA.
 * This is deliberately designed to use few C++ features in order to work with most
 * CUDA SDK versions. It is friendlier to use than the cuComplex type because it
 * provides more operator overloads.
 * The class should work in host code and in device code and also in emulation mode.
 * Also this has been tested on any OS that the CUDA SDK is available for.
 */
template <typename T2, typename T> struct /*ALIGN(8)*/ _cudacomplex
{
    typedef T2 ComplexType;
    typedef T AtomType;

    // float2  is a native CUDA type and allows for coalesced 128 byte access
    // double2 is a native CUDA type and allows for coalesced 256 byte access
    // when accessed according to CUDA's memory coalescing rules.
    // x member is real component
    // y member is imaginary component
    T2 value;

    // ctor
    _cudacomplex(const T a = 0.0, const T b = 0.0) {
	value.x = a; value.y = b;
    };

    // assignment of a scalar to complex
    _cudacomplex<T2, T>& operator=(const T REF(a)) {
	value.x = a; value.y = 0;
	return *this;
    };

    // assignment of a pair of Ts to complex
    _cudacomplex<T2, T>& operator=(const T ARRAYREF(a,2)) {
	value.x = a[0]; value.y = a[1];
	return *this;
    };

    // return references to the real and imaginary components
    M_HOSTDEVICE T& real() {return value.x;};
    M_HOSTDEVICE T& imag() {return value.y;};

    // const versions
    M_HOSTDEVICE T real() const {return value.x;};
    M_HOSTDEVICE T imag() const {return value.y;};

    // add complex numbers
    M_HOSTDEVICE _cudacomplex<T2, T> operator+(const _cudacomplex<T2, T> REF(b)) const {
        _cudacomplex<T2, T> result( value.x + b.value.x, value.y  + b.value.y );
        return result;
    }

    // append complex numbers
    M_HOSTDEVICE _cudacomplex<T2, T> operator+=(const _cudacomplex<T2, T> REF(b)) {
	value.x += b.value.x;
	value.y += b.value.y;
        return *this;
    }

    // add scalar to complex
    M_HOSTDEVICE _cudacomplex<T2, T> operator+(const T REF(b)) const {
        _cudacomplex<T2, T> result( value.x + b, value.y );
        return result;
    }

    // subtract complex numbers
    M_HOSTDEVICE _cudacomplex<T2, T> operator-(const _cudacomplex<T2, T> REF(b)) const {
        _cudacomplex<T2, T> result( value.x - b.value.x, value.y  - b.value.y );
        return result;
    }

    // negate a complex number
    M_HOSTDEVICE _cudacomplex<T2, T> operator-() const {
        _cudacomplex<T2, T> result( -value.x, -value.y );
        return result;
    }

    // subtract scalar from complex
    M_HOSTDEVICE _cudacomplex<T2, T> operator-(const T REF(b)) const {
	_cudacomplex<T2, T> result( value.x - b, value.y );
	return result;
    }

    // multiply complex numbers
    M_HOSTDEVICE _cudacomplex<T2, T> operator*(const _cudacomplex<T2, T> REF(b)) const {
        _cudacomplex<T2, T> result( value.x * b.value.x - value.y * b.value.y,
				    value.y * b.value.x + value.x * b.value.y );
        return result;
    }

    // multiply complex with scalar
    M_HOSTDEVICE _cudacomplex<T2, T> operator*(const T REF(b)) const {
	_cudacomplex<T2, T> result( value.x * b, value.y * b );
	return result;
    }

    // divide complex numbers
    M_HOSTDEVICE _cudacomplex<T2, T> operator/(const _cudacomplex<T2, T> REF(b)) const {
        T tmp = ( b.value.x * b.value.x + b.value.y * b.value.y );
        _cudacomplex<T2, T> result( (value.x * b.value.x + value.y * b.value.y ) / tmp,
				    (value.y * b.value.x - value.x * b.value.y ) / tmp );
        return result;
    }

    // divide complex by scalar
    M_HOSTDEVICE _cudacomplex<T2, T> operator/(const T REF(b)) const {
	_cudacomplex<T2, T> result( value.x / b, value.y / b );
	return result;
    }

    // complex conjugate
    M_HOSTDEVICE _cudacomplex<T2, T> operator~() const {
	_cudacomplex<T2, T> result( value.x, -value.y );
	return result;
    }

    // complex conjugate
    M_HOSTDEVICE bool operator==(const T REF(a)) const { return value.x == a && value.y == 0; }

    // complex modulus (complex absolute)
    M_HOSTDEVICE T abs() const {
	T result = sqrt( value.x*value.x + value.y*value.y );
	return result;
    }

    // complex phase angle
    M_HOSTDEVICE _cudacomplex<T2, T> phase() {
	T result = __atan2( value.y, value.x );
	return result;
    }

    // a possible alternative to a _cudacomplex constructor
    static M_HOSTDEVICE _cudacomplex<T2, T> make_cudacomplex(T a, T b)
    {
        _cudacomplex<T2, T> res;
        res.real() = a;
        res.imag() = b;
        return res;
    }

    // return constant number one
    static M_HOSTDEVICE /*const*/ _cudacomplex<T2, T> one() {
	return make_cudacomplex((T)1.0, (T)0.0);
    }

    // return constant number zero
    static M_HOSTDEVICE const _cudacomplex<T2, T> zero() {
	return make_cudacomplex((T)0.0, (T)0.0);
    }

    // return constant number I
    static M_HOSTDEVICE const _cudacomplex<T2, T> I() {
	return make_cudacomplex((T)0.0, (T)1.0);
    }
};


//
// Define the common complex types
//
typedef _cudacomplex<float2, float> singlecomplex;
typedef _cudacomplex<double2, double> doublecomplex;
//typedef _cudacomplex<doublesingle2, doublesingle> doublesinglecomplex;


//
// Non-member overloads for single complex
//

// subtract single complex from scalar
HOSTDEVICE _cudacomplex<float2, float> operator-(const float REF(a), const _cudacomplex<float2, float> REF(b)) {
    _cudacomplex<float2, float> result( a - b.value.x, -b.value.y );
    return result;
}

// add single complex to scalar
HOSTDEVICE _cudacomplex<float2, float> operator+(const float REF(a), const _cudacomplex<float2, float> REF(b)) {
    _cudacomplex<float2, float> result( a + b.value.x, b.value.y );
    return result;
}

// multiply scalar with single complex
HOSTDEVICE _cudacomplex<float2, float> operator*(const float REF(a), const _cudacomplex<float2, float> REF(b)) {
    _cudacomplex<float2, float> result( a * b.value.x, a * b.value.y );
    return result;
}

// divide scalar by single complex
HOSTDEVICE _cudacomplex<float2, float> operator/(const float REF(a), const _cudacomplex<float2, float> REF(b)) {
    float tmp = ( b.value.x * b.value.x + b.value.y * b.value.y );
    _cudacomplex<float2, float> result( ( a * b.value.x ) / tmp, ( -a * b.value.y ) / tmp );
    return result;
}

// in order to compare two complex types
HOSTDEVICE bool operator<(const _cudacomplex<float2, float> REF(a), const _cudacomplex<float2, float> REF(b))
{
    return a.real() < b.real();
}


//
// Non-member overloads for double complex
//

// subtract double complex from scalar
HOSTDEVICE _cudacomplex<double2, double> operator-(const double REF(a), const _cudacomplex<double2, double> REF(b)) {
    _cudacomplex<double2, double> result( a - b.value.x, -b.value.y );
    return result;
}

// add double complex to scalar
HOSTDEVICE _cudacomplex<double2, double> operator+(const double REF(a), const _cudacomplex<double2, double> REF(b)) {
    _cudacomplex<double2, double> result( a + b.value.x, b.value.y );
    return result;
}

// multiply scalar with double complex
HOSTDEVICE _cudacomplex<double2, double> operator*(const double REF(a), const _cudacomplex<double2, double> REF(b)) {
    _cudacomplex<double2, double> result( a * b.value.x, a * b.value.y );
    return result;
}

// divide scalar by double complex
HOSTDEVICE _cudacomplex<double2, double> operator/(const double REF(a), const _cudacomplex<double2, double> REF(b)) {
    double tmp = ( b.value.x * b.value.x + b.value.y * b.value.y );
    _cudacomplex<double2, double> result( ( a * b.value.x ) / tmp, ( -a * b.value.y ) / tmp );
    return result;
}

// in order to compare two complex types
HOSTDEVICE bool operator<(const _cudacomplex<double2, double> REF(a), const _cudacomplex<double2, double> REF(b))
{
    return a.real() < b.real();
}

// a possible alternative to a single complex constructor
HOSTDEVICE singlecomplex make_singlecomplex(float a, float b)
{
    singlecomplex res;
    res.real() = a;
    res.imag() = b;
    return res;
}


// a possible alternative to a double complex constructor
HOSTDEVICE doublecomplex make_doublecomplex(double a, double b)
{
    doublecomplex res;
    res.real() = a;
    res.imag() = b;
    return res;
}

template <typename T2, typename T>
T abs( const _cudacomplex< T2, T > REF(z) ) { return z.abs(); }

template <typename T2, typename T>
_cudacomplex< T2, T > sqrt( const _cudacomplex< T2, T > REF(z) )
{
    T x = z.real();
    T y = z.imag();

    if (x == T())
	{
	    T t = sqrt( abs(y) / 2 );
	    return _cudacomplex< T2, T >( t, y < T() ? -t : t );
	}
    else
	{
	    T t = sqrt( 2 * ( abs(z) + abs(x) ) );
	    T u = t / 2;
	    return x > T()
		? _cudacomplex< T2, T >( u, y / t )
		: _cudacomplex< T2, T >( abs(y) / t, y < T() ? -u : u );
	}
}

#endif // !CUDACOMPLEX_H
