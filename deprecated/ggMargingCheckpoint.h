// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// ggMargingCheckpoint.h
/*
  Contact: Caner Candan <caner@candan.fr>
*/
//-----------------------------------------------------------------------------

#ifndef _ggMargingCheckpoint_h
#define _ggMargingCheckpoint_h

/*
  ggMargingCheckpoint used to add some routines around the computation's call.

  @ingroup checkpointing
*/
class ggMargingCheckpoint : public ggMarging
{
public:
    void add( const ggMarging& checkpoint )
    {
	_checkpoints.push_back( checkpoint );
    }

    void pre()
    {
	std::vector< const ggMarging& >::const_iterator it = _checkpoints.begin();
	std::vector< const ggMarging& >::const_iterator end = _checkpoints.end();
	for ( ; it != end; ++it ) { (*it).pre(); }
    }

    void post()
    {
	std::vector< const ggMarging& >::const_iterator it = _checkpoints.begin();
	std::vector< const ggMarging& >::const_iterator end = _checkpoints.end();
	for ( ; it != end; ++it ) { (*it).post(); }
    }

private:
    std::vector< const ggMarging& > _checkpoints;
};

#endif // !_ggMargingCheckpoint_h
