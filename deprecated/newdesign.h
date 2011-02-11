// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// newdesign.h

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

/*
  This files provides all the following classes:

  - Printable
  - Object
   - Vector
    - Matrix

   - Functor
    - UnaryFunc -> std::unary_function
    - ConstUnaryFunc
    - BinaryFunc -> std::binary_function
    - ConstBinaryFunc
    - ComputeFunc

   - ConstBinaryFunc
    - Addition
    - Multiply
    - Sub
    - Division

   - PolynomeFunc
    - Axpy

   - UnaryFunc
    - Normalize

   - ComputeFunc
    - Algo
     - EasyLA
    - Solver
     - KrylovCG1
     - KrylovCG2
     - KrylovORTH
    - Checker

   - Continue
    - MargingContinue
     - MargingCheckpoint
     - MargingMonitor
      - Timer

    - SolverContinue
     - SolverCheckpoint
      - SolverMonitor
      - FileSnapshot
      - Stdout
 */

/*
  Example of code usage in using the new design:

  int main()
  {
      TimerCUDA timer;
      MargingCheckpoint m_checkpoint;
      m_checkpoint.add( timer );
      Stdout out;
      SolverCheckpoint s_checkpoint;
      s_checkpoint.add( out );

      MultiplyMatVecCUDA mm;
      AdditionVectorCUDA add;

      KrylovCG1 solver( mm, add, s_checkpoint );

      EasyLA algo( solver, m_checkpoint );
      algo( M );
  }
*/
//-----------------------------------------------------------------------------

#ifndef _newdesign_h
#define _newdesign_h

#include <functional> // std::unary_function, std::binary_function
#include <iostream> // std::ostream
#include <vector>

#include "Param.h"

namespace LA
{
    /*
      Object used to be the base class for the whole hierarchy.

      @ingroup Core
    */
    class Object
    {
    public:
	/// Virtual dtor needed in virtual class hierarchies.
	virtual ~Object() {}

	/** Virtual methode to return class id. It should be redefined in the derivated class.
	 */
	virtual std::string className() const = 0;
    };

    /* All classes implementing Printable must define printOn
       function. This allows to print some information about
       the class's data.
    */
    class Printable
    {
    public:
	virtual ~Printable() {}

	virtual void printOn( std::ostream& ) const = 0;
    };

    std::ostream& operator<<( std::ostream& os, const Printable& o )
    {
	o.printOn( os );
	return os;
    }

    /* this is our base class for all operator classes */
    class Functor : public Object {};

    /* here's an overload of the STL class unary_function for our unary computing operators
       there is a virtual pure method that all derivated classes must define:
       virtual Result operator()( Arg1 ) = 0;
    */
    template < typename Arg1, typename Result >
    class UnaryFunc : public Functor, public std::unary_function< Arg1, Result > {};

    /* the const version */
    template < typename Arg1, typename Result >
    class ConstUnaryFunc : public Functor
    {
    public:
	virtual ~ConstUnaryFunc() {}
	virtual Result operator()( const Arg1& ) const = 0;
    };

    /* here's an overload of the STL class binary_function for our binary computing operators
       there is a virtual pure method that all derivated classes must define:
       virtual Result operator()( Arg1, Arg2 ) = 0;
    */
    template < typename Arg1, typename Arg2, typename Result >
    class BinaryFunc : public Functor, public std::binary_function< Arg1, Arg2, Result > {};

    /* the const version */
    template < typename Arg1, typename Arg2, typename Result >
    class ConstBinaryFunc : public Functor
    {
    public:
	virtual ~ConstBinaryFunc() {}
	virtual Result operator()( const Arg1&, const Arg2& ) const = 0;
    };

    /* the functor with no argument only a result type */
    template < typename Result >
    class ComputeFunc : public Functor
    {
    public:
	virtual ~ComputeFunc() {}
	virtual Result operator()() const = 0;
    };

    /* the polynome functor */
    template < typename Arg1, typename Arg2, typename Arg3, typename Result >
    class PolynomeFunc : public Functor
    {
    public:
	virtual ~PolynomeFunc() {}
	virtual Result operator()( const Arg1& , const Arg2&, const Arg3& ) const = 0;
    };

    /* the axpy functor */
    template < typename A, typename x >
    class Axpy : public PolynomeFunc< A, x, x, x >
    {
    public:
	x operator()( const A& a, const x& b, const x& c ) const { return a * b + c; }
    };

    /* here's the nomalization base class */
    template < typename LAT >
    class Normalize : public ConstUnaryFunc< LAT, LAT > {};

    /* here's the addition */
    template < typename Arg1, typename Arg2, typename Result >
    class Addition : public ConstBinaryFunc< Arg1, Arg2, Result >
    {
    public:
	Result operator()( const Arg1& a, const Arg2& b) const { return a + b; }
    };

    /* here's the multiplication */
    template < typename Arg1, typename Arg2, typename Result >
    class Multiply : public ConstBinaryFunc< Arg1, Arg2, Result >
    {
    public:
	Result operator()( const Arg1& a, const Arg2& b) const { return a * b; }
    };

    /* here's the subtraction */
    template < typename Arg1, typename Arg2, typename Result >
    class Subtraction : public ConstBinaryFunc< Arg1, Arg2, Result >
    {
    public:
	Result operator()( const Arg1& a, const Arg2& b) const { return a - b; }
    };

    /* here's the division */
    template < typename Arg1, typename Arg2, typename Result >
    class Division : public ConstBinaryFunc< Arg1, Arg2, Result >
    {
    public:
	Result operator()( const Arg1& a, const Arg2& b) const { return a / b; }
    };

    /* here's the solver base class */
    template < typename LAT >
    class Solver : public ComputeFunc< LAT > {};

    /* here's the krylovcg */
    template < typename LAT >
    class KrylovCG : public Solver< LAT >
    {
    public:
	LAT operator()() const
	{
	    // TODO
	    return LAT();
	}
    };

    /* here's the checker base class */
    template < typename LAT >
    class Checker : public ComputeFunc< LAT > {};

    /* here's the algorithm base class */
    template < typename LAT >
    class Algo : public ComputeFunc< LAT > {};

    /* here's the easyla */
    template < typename LAT >
    class EasyLA : public Algo< LAT >
    {
    public:
	EasyLA( Solver< LAT >& solver ) : _solver( solver ) {}

    private:
	Solver< LAT >& _solver;
    };

    /*
      Continue used to add some routines around the computation's call.

      @ingroup continuator
    */
    template < typename LAT >
    class Continue : public Object {};

    /*
      MargingContinue used to add some routines around the computation's call.

      @ingroup continuator
    */
    template < typename LAT >
    class MargingContinue : public Continue< LAT >
    {
    public:
	virtual void pre() const = 0;
	virtual void post() const = 0;
    };

    /*
      MargingCheckpoint used to add some routines around the computation's call.

      @ingroup checkpointing
    */
    template < typename LAT >
    class MargingCheckpoint : public MargingContinue< LAT >
    {
    public:
 	void add( const MargingContinue< LAT >& continuator ) { _continuators.push_back( &continuator ); }

	void pre() const
	{
	    for ( size_t i = 0, size = _continuators.size(); i < size; ++i ) { _continuators[i]->pre(); }
	}

	void post() const
	{
	    for ( size_t i = 0, size = _continuators.size(); i < size; ++i ) { _continuators[i]->post(); }
	}

    private:
	std::vector< MargingContinue< LAT >* > _continuators;
    };

    /*
      SolverContinue used to add some routines around the solver's call.

      @ingroup continuator
    */
    template < typename LAT >
    class SolverContinue : public Continue< LAT >, public ConstUnaryFunc< int, void > {};

    /* Monitor class for solver */
    class SolverMonitor : public ComputeFunc< SolverMonitor& >
    {};

    /*
      SolverCheckpoint used to add some routines around the solver's call.

      @ingroup checkpointing
    */
    template < typename LAT >
    class SolverCheckpoint : public SolverContinue< LAT >
    {
    public:
 	void add( const SolverContinue< LAT >& continuator ) { _continuators.push_back( &continuator ); }

 	void add( const SolverMonitor& monitor ) { _monitors.push_back( &monitor ); }

	void operator()( int ) const
	{
	    for ( size_t i = 0, size = _continuators.size(); i < size; ++i ) { (*_continuators[i])(); }
	}

    private:
	std::vector< SolverContinue< LAT >* > _continuators;
	std::vector< SolverMonitor* > _monitors;
    };

}

#endif // !_newdesign_h
