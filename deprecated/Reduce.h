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

#ifndef _Reduce_h
#define _Reduce_h

template < typename Atom >
class Reduce
{
public:
    __global__ void operator()( Atom* in, Atom* out ) = 0;
};

template < typename Atom >
class CudaReduce : public Reduce< Atom >
{
public:
    CudaReduce( CudaDot* dot ) : _dot(dot) {}

    __global__ void operator()( Atom* in, Atom* out )
    {
	int glo_id = blockIdx.x * blockDim.x + threadIdx.x;
	int loc_id = threadIdx.x;

	__shared__ float s_in[blockDim.x];

	__syncthreads();

	for(int s = blockDim.x/2 ; s > 0 ; s /= 2)
	    {
		if (loc_id < s)
		    {
			s_in[loc_id] += s_in[loc_id + s];
		    }
		__syncthreads();
	    }

	if (loc_id == 0)
	    {
		out[blockIdx.x] = s_in[loc_id];
	    }
	__syncthreads();
    }
};

class Dot {};
class CudaDot {};

#endif // !_Reduce_h
