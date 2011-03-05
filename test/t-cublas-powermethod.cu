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

#include <linear_algebra/detail/device/cublas/cublas>

using namespace linear_algebra::detail::device::cublas;
using namespace core_library;

typedef float T;
//typedef double T;

int main(int ac, char** av)
{
    Parser parser(ac, av);

    int size = parser.createParam(5, "matrixSize", "The size of the matrix", 'm').value();

    make_help(parser);

    logger << "Matrix size: " << size << std::endl;

    T* hA = new T[size*size];
    T* hx = new T[size];

    for (int i = 0; i < size; i++)
	{
	    for (int j = 0; j < size; j++)
		{
		    hA[i * size + j] = 1.0/(((T)(i+1))+((T)(j+1))-1.0);
		}
	}

    hx[0] = 1;
    for(int i = 1; i < size; i++) { hx[i] = 0; }

    Matrix<T> A(size, size);
    Vector<T> x(size);

    A = hA;
    x = hx;

    delete[] hA;
    delete[] hx;

    IterContinue<T> iter(100);
    TolContinue<T> tol(1e5);
    CheckPoint<T> checkpoint(iter);
    checkpoint.add(tol);

    TimeCounter counter;
    checkpoint.add(counter);

    StdoutMonitor monitor;
    monitor.add(counter);
    checkpoint.add(monitor);

    PowerMethod<T> pm(checkpoint);

    logger << "eigenvalue: " << pm( A, x ) << std::endl;

    return 0;
}
