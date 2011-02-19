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

#ifndef _linear_algebra_detail_device_cublas_Vector_h
#define _linear_algebra_detail_device_cublas_Vector_h

#include <cublas.h>

#include <stdexcept>
#include <algorithm>

#include <linear_algebra/Vector.h>

namespace linear_algebra
{
    namespace detail
    {
	namespace device
	{
	    namespace cublas
	    {

		template < typename Atom >
		class Vector : public linear_algebra::Vector< Atom >
		{
		public:
		    Vector() : _deviceVector(NULL), _n(0)
		    {}

		    Vector(int n) : _deviceVector(NULL), _n(n)
		    {
			createDeviceVector(_deviceVector, _n);
		    }

		    Vector(int n, Atom value) : _deviceVector(NULL), _n(n)
		    {
			Atom* hostVector;
			createHostVector(hostVector, _n);
			fillHostVector(hostVector, _n, value);
			createDeviceVector(_deviceVector, _n);
			memcpyHostToDevice(hostVector, _deviceVector, _n);
			destroyHostVector(hostVector);
		    }

		    ~Vector()
		    {
			if ( !_deviceVector ) { return; }
			destroyDeviceVector(_deviceVector);
		    }

		    Vector& operator=( Atom*& v )
		    {
			if ( !_deviceVector ) { return *this; }
			memcpyHostToDevice(v, _deviceVector, _n);
			return *this;
		    }

		    std::string className() const { return "Vector"; }

		    virtual void printOn(std::ostream& _os) const
		    {
			if ( !_deviceVector ) { return; }
			if ( _n <= 0 ) { return; }

			Atom* hostVector;
			createHostVector( hostVector, _n );

			cublasStatus stat = cublasGetVector(_n, sizeof(*_deviceVector), _deviceVector, 1, hostVector, 1);
			if ( stat != CUBLAS_STATUS_SUCCESS )
			    {
				throw std::runtime_error("data download failed");
			    }

			//memcpyDeviceToHost( _deviceVector, hostVector );

			_os << "[" << hostVector[0];
			for ( int i = 1; i < _n; ++i )
			    {
				_os << ", " << hostVector[i];
			    }
			_os << "]";

			destroyHostVector(hostVector);
		    }

		    operator Atom*() const
		    {
			if ( !_deviceVector )
			    {
				throw std::runtime_error("deviceVector is not allocated on GPU memory");
			    }
			return _deviceVector;
		    }

		    inline int size() const { return _n; }

		    void resize(int n)
		    {
			if ( _deviceVector )
			    {
				destroyDeviceVector( _deviceVector );
			    }

			_n = n;
			createDeviceVector(_deviceVector, _n);
		    }

		private:
		    Atom* _deviceVector;
		    int _n;

		public:
		    /// Here's some high level cublas routines in static

		    static void createDeviceVector(Atom*& deviceVector, int n)
		    {
			cublasStatus stat = cublasAlloc(n, sizeof(*deviceVector), (void**)&deviceVector);
			if ( stat != CUBLAS_STATUS_SUCCESS )
			    {
				throw std::runtime_error("data memory allocation failed");
			    }
		    }

		    static void destroyDeviceVector(Atom*& deviceVector)
		    {
			cublasFree(deviceVector);
			deviceVector = NULL;
		    }

		    static void createHostVector(Atom*& hostVector, int n)
		    {
			hostVector = (Atom*)malloc(n*sizeof(*hostVector));
			if ( hostVector == NULL )
			    {
				throw std::runtime_error("out of memory");
			    }
		    }

		    static void destroyHostVector(Atom*& hostVector)
		    {
			free(hostVector);
			hostVector = NULL;
		    }

		    static void fillHostVector(Atom*& hostVector, int n, Atom value)
		    {
			for (int i = 0; i < n; ++i)
			    {
				*(hostVector + i) = value;
			    }
		    }

		    static void memcpyHostToDevice(Atom*& hostVector, Atom*& deviceVector, int n)
		    {
			cublasStatus stat = cublasSetVector(n, sizeof(*hostVector), hostVector, 1, deviceVector, 1);
			if ( stat != CUBLAS_STATUS_SUCCESS )
			    {
				throw std::runtime_error("data download failed");
			    }
		    }

		    static void memcpyDeviceToHost(Atom*& deviceVector, Atom*& hostVector, int n)
		    {
			cublasStatus stat = cublasGetVector(n, sizeof(*deviceVector), deviceVector, 1, hostVector, 1);
			if ( stat != CUBLAS_STATUS_SUCCESS )
			    {
				throw std::runtime_error("data download failed");
			    }
		    }

		};
	    }
	}
    }
}

#endif // !_linear_algebra_detail_device_cublas_Vector_h
