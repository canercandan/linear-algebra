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

#ifndef _MultiplyMatVec_h
#define _MultiplyMatVec_h

template < typename Atom >
class MultiplyMatVec
{
public:
    void operator()( Vector< Atom >&, Matrix< Atom >&, Vector< Atom >& ) = 0;
};

template < typename Atom >
class CudaMultiplyMatVec : public MultiplyMatVec< Atom >
{
public:
    __global__ void operator()( Vector< Atom >& y, Matrix< Atom >& A, Vector< Atom >& x )
    {
	// TODO
    }
};

#endif // !_MultiplyMatVec_h
