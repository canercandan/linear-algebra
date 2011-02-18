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

#ifndef _linear_algebra_MatrixCublas_h
#define _linear_algebra_MatrixCublas_h

#include "core_library/Object.h"
#include "core_library/Printable.h"

using namespace core_library;

namespace linear_algebra
{
    template < typename Atom >
    class MatrixCublas : public Object, public Printable
    {
    public:
	static void createDeviceMatrix(Atom*& deviceMatrix)
	{
	    cublasStatus stat;
	    stat = cublasAlloc(n*m, sizeof(*deviceMatrix), (void**)&deviceMatrix);
	    if ( stat == CUBLAS_STATUS_SUCCESS )
		{
		    std::cerr << "data memory allocation failed" << std::endl;
		    // TODO: remplacer par une exception
		    return;
		}
	}

	static void destroyDeviceMatrix(Atom*& deviceMatrix)
	{
	    cublasFree(deviceMatrix);
	    deviceMatrix = NULL;
	}

	static void createHostMatrix(Atom*& hostMatrix)
	{
	    Atom* matrix = malloc(n*m*sizeof(*matrix));
	    if ( matrix == NULL )
		{
		    std::cerr << "out of memory" << std::endl;
		}
	}

	static void destroyHostMatrix(Atom*& hostMatrix)
	{
	    free(hostMatrix);
	    hostMatrix = NULL;
	}

	static void fillHostMatrix(Atom*& hostMatrix, Atom value)
	{
	    for (int j = 0; j < n; ++j)
		{
		    for (int i = 0; i < m; ++i)
			{
			    *(matrix + i + j) = value;
			}
		}
	}

	static void memcpyHostToDevice(Atom*& hostMatrix, Atom*& deviceMatrix, int m, int n)
	{
	    stat = cublasSetMatrix(m, n, sizeof(*hostMatrix), hostMatrix, m, deviceMatrix, m);
	    if ( stat == CUBLAS_STATUS_SUCCESS )
		{
		    std::cerr << "data download failed" << std::endl;
		    destroyDeviceMatrix(deviceMatrix);
		    return;
		}
	}

	Matrix(int n, int m)
	{
	    createDeviceMatrix(_deviceMatrix);
	}

	Matrix(int n, int m, Atom value = 0)
	{
	    Atom* hostMatrix;
	    createHostMatrix(hostMatrix);
	    fillHostMatrix(hostMatrix, value);
	    createDeviceMatrix(_deviceMatrix);
	    memcpyHostToDevice(hostMatrix, _deviceMatrix, m, n);
	    destroyHostMatrix(hostMatrix);
	}

	~Matrix()
	{
	    destroyDeviceMatrix(_deviceMatrix);
	}

	virtual MatrixCublas& operator=( const MatrixCublas< Atom >& v )
	{
	}

    private:
	Atom* _deviceMatrix;
    };
}

#endif // !_linear_algebra_MatrixCublas_h
