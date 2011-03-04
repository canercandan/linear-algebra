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

typedef float T;
//typedef double T;

int main(void)
{
    const int N = 5;
    const int M = 5;

    T* hA = new T[N*M];
    T* hx = new T[N];

    for (int i = 0; i < N; i++)
	{
	    for (int j = 0; j < M; j++)
		{
		    hA[i * N + j] = 1.0/(((T)(i+1))+((T)(j+1))-1.0);
		}
	}

    hx[0] = 1;
    for(int i = 1; i < N; i++) { hx[i] = 0; }

    Matrix<T> A(M, N);
    Vector<T> x(N);

    A = hA;
    x = hx;

    delete[] hA;
    delete[] hx;

    core_library::IterContinue<T> iter(100);
    core_library::TolContinue<T> tol(1e5);
    core_library::CheckPoint<T> checkpoint(iter);
    checkpoint.add(tol);

    PowerMethod<T> pm(checkpoint);

    core_library::logger << "eigenvalue: " << pm( A, x ) << std::endl;

    return 0;
}
