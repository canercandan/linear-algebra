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

#ifndef _Scal_h
#define _Scal_h

template < typename Atom >
class Scal
{
public:
    __global__ void operator()( Atom* in, Atom* out ) = 0;
};

template < typename Atom >
class CudaScal : public Scal< Atom >
{
public:
    __global__ void operator()( Vector< Atom >& v, Scalar< Atom > eigenvalue )
    {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if ( i >= v.size() ) { return; }
	v[i] /= eigenvalue;
    }
};

class Dot {};
class CudaDot {};

#endif // !_Scal_h
