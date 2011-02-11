#ifndef _ggPrintable_h
#define _ggPrintable_h

/* all classes implementing ggPrintable must define printOn
   function. This allows to print some information about
   the class's data.
 */
class ggPrintable
{
public:
    virtual ~ggPrintable() {}

    virtual void printOn( std::ostream& ) const = 0;
};

std::ostream& operator<<( std::ostream& os, const ggPrintable& o )
{
    o.printOn( os );
    return os;
}

#endif // !_ggPrintable_h
