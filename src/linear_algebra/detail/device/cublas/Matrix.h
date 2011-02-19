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

#ifndef _linear_algebra_detail_device_cublas_Matrix_h
#define _linear_algebra_detail_device_cublas_Matrix_h

#include <cublas.h>

#include <stdexcept>

#include <linear_algebra/Matrix.h>

namespace linear_algebra
{
    namespace detail
    {
	namespace device
	{
	    namespace cublas
	    {

		template < typename Atom >
		class Matrix : public linear_algebra::Matrix< Atom >
		{
		public:
		    Matrix(int n) : _deviceMatrix(NULL), _hostMatrix(NULL), _n(n), _m(n)
		    {
			createDeviceMatrix(_deviceMatrix, _n, _m);
			createHostMatrix(_hostMatrix, _n, _m);
		    }

		    Matrix(int n, int m) : _deviceMatrix(NULL), _hostMatrix(NULL), _n(n), _m(m)
		    {
			createDeviceMatrix(_deviceMatrix, _n, _m);
			createHostMatrix(_hostMatrix, _n, _m);
		    }

		    Matrix(int n, int m, Atom value) : _deviceMatrix(NULL), _hostMatrix(NULL), _n(n), _m(m)
		    {
			createHostMatrix(_hostMatrix, _n, _m);
			fillHostMatrix(_hostMatrix, value, _n, _m);
			createDeviceMatrix(_deviceMatrix, _n, _m);
			memcpyHostToDevice(_hostMatrix, _deviceMatrix, _n, _m);
		    }

		    ~Matrix()
		    {
			destroyDeviceMatrix(_deviceMatrix);
			destroyHostMatrix(_hostMatrix);
		    }

		    Matrix& operator=( Atom*& m )
		    {
			memcpyHostToDevice(m, _deviceMatrix, _n, _m);
			return *this;
		    }

		    std::string className() const { return "Matrix"; }

		    virtual void printOn(std::ostream& _os) const
		    {
			// TODO
		    }

		    operator Atom*() const { return _deviceMatrix; }
		    operator Atom*() { return _deviceMatrix; }

		    inline int rows() const { return _n; }
		    inline int cols() const { return _m; }

		private:
		    Atom* _deviceMatrix;
		    Atom* _hostMatrix;
		    int _n;
		    int _m;

		public:
		    /// Here's some high level cublas routines in static

		    static void createDeviceMatrix(Atom*& deviceMatrix, int n, int m)
		    {
			cublasStatus stat = cublasAlloc(n*m, sizeof(*deviceMatrix), (void**)&deviceMatrix);
			if ( stat != CUBLAS_STATUS_SUCCESS )
			    {
				throw std::runtime_error("data memory allocation failed");
			    }
		    }

		    static void destroyDeviceMatrix(Atom*& deviceMatrix)
		    {
			cublasFree(deviceMatrix);
			deviceMatrix = NULL;
		    }

		    static void createHostMatrix(Atom*& hostMatrix, int n, int m)
		    {
			hostMatrix = (Atom*)malloc(n*m*sizeof(*hostMatrix));
			if ( hostMatrix == NULL )
			    {
				throw std::runtime_error("out of memory");
			    }
		    }

		    static void destroyHostMatrix(Atom*& hostMatrix)
		    {
			free(hostMatrix);
			hostMatrix = NULL;
		    }

		    static void fillHostMatrix(Atom*& hostMatrix, int n, int m, Atom value)
		    {
			for (int j = 0; j < n; ++j)
			    {
				for (int i = 0; i < m; ++i)
				    {
					*(hostMatrix + i + j) = value;
				    }
			    }
		    }

		    static void memcpyHostToDevice(Atom*& hostMatrix, Atom*& deviceMatrix, int n, int m)
		    {
			cublasStatus stat = cublasSetVector(m*n, sizeof(*hostMatrix), hostMatrix, 1, deviceMatrix, 1);
			if ( stat != CUBLAS_STATUS_SUCCESS )
			    {
				throw std::runtime_error("data download failed");
			    }
		    }

		    static void memcpyDeviceToHost(Atom*& deviceMatrix, Atom*& hostMatrix, int n, int m)
		    {
			cublasStatus stat = cublasGetVector(m*n, sizeof(*deviceMatrix), deviceMatrix, 1, hostMatrix, 1);
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

#endif // !_linear_algebra_detail_device_cublas_Matrix_h
