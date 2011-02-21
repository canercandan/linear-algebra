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

#ifndef _linear_algebra_detail_device_cublas_Array_h
#define _linear_algebra_detail_device_cublas_Array_h

#include <cstdlib>
#include <stdexcept>

#include <cublas.h>

#include <linear_algebra/Array.h>

#include "common.h"

namespace linear_algebra
{
    namespace detail
    {
	namespace device
	{
	    namespace cublas
	    {
		template < typename Atom >
		class Array : virtual public linear_algebra::Array< Atom >
		{
		public:
		    Array() : _deviceArray(NULL), _size(0) {}

		    Array(int n) : _deviceArray(NULL), _size(n)
		    {
			createDeviceArray(_deviceArray, _size);
		    }

		    Array(Atom value) : _deviceArray(NULL), _size(1)
		    {
			createDeviceArray(_deviceArray, _size);
			memcpyHostToDevice(&value, _deviceArray, _size);
		    }

		    Array(int n, Atom value) : _deviceArray(NULL), _size(n)
		    {
			Atom* hostArray;
			createHostArray(hostArray, _size);
			fillHostArray(hostArray, _size, value);
			createDeviceArray(_deviceArray, _size);
			memcpyHostToDevice(hostArray, _deviceArray, _size);
			destroyHostArray(hostArray);
		    }

		    ~Array()
		    {
			if ( !_deviceArray ) { return; }
			destroyDeviceArray(_deviceArray);
		    }

		    Array& operator=( Atom*& data )
		    {
			if ( !_deviceArray )
			    {
				throw std::runtime_error("deviceArray is not allocated on GPU memory");
			    }
			memcpyHostToDevice(data, _deviceArray, _size);
			return *this;
		    }

		    virtual std::string className() const { return "Array"; }

		    virtual void printOn(std::ostream& _os) const
		    {
			if ( !_deviceArray ) { return; }
			if ( _size <= 0 ) { return; }

			Atom* hostArray;
			createHostArray( hostArray, _size );

			CUBLAS_CALL( cublasGetVector(_size, sizeof(*_deviceArray), _deviceArray, 1, hostArray, 1) );

			_os << "[" << hostArray[0];
			for ( int i = 1; i < _size; ++i )
			    {
				_os << ", " << hostArray[i];
			    }
			_os << "]";

			destroyHostArray(hostArray);
		    }

		    operator Atom*() const
		    {
			if ( !_deviceArray )
			    {
				throw std::runtime_error("deviceArray is not allocated on GPU memory");
			    }
			return _deviceArray;
		    }

		    inline int size() const { return _size; }

		    void resize(int size)
		    {
			if ( _deviceArray )
			    {
				destroyDeviceArray( _deviceArray );
			    }

			_size = size;
			createDeviceArray(_deviceArray, _size);
		    }

		protected:
		    Atom* _deviceArray;
		    int _size;

		public:
		    /// Here's some high level cublas routines in static

		    static void createDeviceArray(Atom*& deviceArray, int n)
		    {
			CUBLAS_CALL( cublasAlloc(n, sizeof(*deviceArray), (void**)&deviceArray) );
		    }

		    static void destroyDeviceArray(Atom*& deviceArray)
		    {
			CUBLAS_CALL( cublasFree(deviceArray) );
			deviceArray = NULL;
		    }

		    static void createHostArray(Atom*& hostArray, int n)
		    {
			hostArray = (Atom*)malloc(n*sizeof(*hostArray));
			if ( hostArray == NULL )
			    {
				throw std::runtime_error("out of memory");
			    }
		    }

		    static void destroyHostArray(Atom*& hostArray)
		    {
			free(hostArray);
			hostArray = NULL;
		    }

		    static void fillHostArray(Atom*& hostArray, int n, Atom value)
		    {
			for (int i = 0; i < n; ++i)
			    {
				*(hostArray + i) = value;
			    }
		    }

		    static void memcpyHostToDevice(Atom*& hostArray, Atom*& deviceArray, int n)
		    {
			CUBLAS_CALL( cublasSetVector(n, sizeof(*hostArray), hostArray, 1, deviceArray, 1) );
		    }

		    static void memcpyDeviceToHost(Atom*& deviceArray, Atom*& hostArray, int n)
		    {
			CUBLAS_CALL( cublasGetVector(n, sizeof(*deviceArray), deviceArray, 1, hostArray, 1) );
		    }

		};
	    }
	}
    }
}

#endif // !_linear_algebra_detail_device_cublas_Array_h
