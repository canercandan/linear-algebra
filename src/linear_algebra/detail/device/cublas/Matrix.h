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

#ifndef _linear_algebra_cublas_Matrix_h
#define _linear_algebra_cublas_Matrix_h

#include "core_library/Object.h"
#include "core_library/Printable.h"

#include <cublas.h>

#include <stdexcept>

namespace linear_algebra
{
    namespace detail
    {
	namespace device
	{
	    namespace cublas
	    {
		template < typename Atom >
		class Matrix : public core_library::Object, public core_library::Printable
		{
		public:
		    static void createDeviceMatrix(Atom*& deviceMatrix, int n, int m)
		    {
			cublasStatus stat = cublasAlloc(n*m, sizeof(*deviceMatrix), (void**)&deviceMatrix);
			if ( stat == CUBLAS_STATUS_SUCCESS )
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
			cublasStatus stat = cublasSetMatrix(m, n, sizeof(*hostMatrix), hostMatrix, m, deviceMatrix, m);
			if ( stat == CUBLAS_STATUS_SUCCESS )
			    {
				destroyDeviceMatrix(deviceMatrix);
				throw std::runtime_error("data download failed");
			    }
		    }

		    Matrix(int n, int m)
		    {
			createDeviceMatrix(_deviceMatrix, n, m);
		    }

		    Matrix(int n, int m, Atom value = 0)
		    {
			Atom* hostMatrix;
			createHostMatrix(hostMatrix, n, m);
			fillHostMatrix(hostMatrix, value, n, m);
			createDeviceMatrix(_deviceMatrix, n, m);
			memcpyHostToDevice(hostMatrix, _deviceMatrix, n, m);
			destroyHostMatrix(hostMatrix);
		    }

		    ~Matrix()
		    {
			destroyDeviceMatrix(_deviceMatrix);
		    }

		    virtual Matrix& operator=( const Matrix< Atom >& v )
		    {
			return *this;
		    }

		private:
		    Atom* _deviceMatrix;
		};
	    }
	}
    }
}

#endif // !_linear_algebra_cublas_Matrix_h
