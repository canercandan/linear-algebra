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
    ParserLogger parser(ac, av);

    int size = parser.createParam(1000, "matrixSize", "The size of the matrix", 'm').value();

    int iterMax = parser.createParam(100, "iter-max", "The maximum of iterations", 'i').value();

    T tol = parser.createParam(1e-5, "tol", "The tolerance of eigenvalue", 't').value();

    make_help(parser);
    make_verbose(parser);

    logger << "Matrix size: " << size << std::endl;

    T* hA = new T[size*size];
    T* hx = new T[size];

    for (int i = 0; i < size; i++)
	{
	    for (int j = 0; j < size; j++)
		{
		    hA[i * size + j] = 1.0/((T(i+1))+(T(j+1))-1.0);
		}
	}

    hx[0] = 1;
    for (int i = 1; i < size; i++) { hx[i] = 0; }

    Matrix<T> A(size, size);
    Vector<T> x(size);

    A = hA;
    x = hx;

    delete[] hA;
    delete[] hx;

    IterContinue<T> iter_cont(iterMax);
    TolContinue<T> tol_cont(tol);
    CombinedContinue<T> continuators(iter_cont);
    continuators.add(tol_cont);

    CheckPoint<T> checkpoint(continuators);

    TimeCounter counter;
    checkpoint.add(counter);

    linear_algebra::PowerMethodStat<T> pm_stat;
    checkpoint.add(pm_stat);

    StdoutMonitor monitor;
    monitor.add(counter);
    monitor.add(pm_stat);

    checkpoint.add(monitor);

    PowerMethod<T> pm(checkpoint);

    logger << "eigenvalue: " << pm( A, x ) << std::endl;

    return 0;
}
