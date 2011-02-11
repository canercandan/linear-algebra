// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// ggObject.h
/*
  Contact: Caner Candan <caner@candan.fr>
*/
//-----------------------------------------------------------------------------

#ifndef _ggObject
#define _ggObject

/*
  ggObject used to be the base class for the whole hierarchy.

  @ingroup Core
*/
class ggObject
{
public:
    /// Virtual dtor needed in virtual class hierarchies.
    virtual ~ggObject() {}

    /** Virtual methode to return class id. It should be redefined in the derivated class.
     */
    virtual std::string className() const = 0;
};

#endif // _ggObject
