// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

/*
(c) Thales group, 2010

    This library is free software; you can redistribute it and/or
    modify it under the terms of the GNU Lesser General Public
    License as published by the Free Software Foundation;
    version 2 of the License.

    This library is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
    Lesser General Public License for more details.

    You should have received a copy of the GNU Lesser General Public
    License along with this library; if not, write to the Free Software
    Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA

Authors:
	Johann Dréo <johann.dreo@thalesgroup.com>
	Caner Candan <caner.candan@thalesgroup.com>
*/

/** @defgroup Logging Logging
 * @ingroup Utilities

 Global logger for EO.

 Here's an example explaning how to use Logger:
 \code
 #include <iostream>
 #include <utils/Logger.h>
 #include <utils/ParserLogger.h>

 int	main(int ac, char** av)
 {
 // We are declaring first an overload of Parser class using Logger
 // component.
 ParserLogger	parser(ac, av);

 // This call is important to allow -v parameter to change user level.
 make_verbose(parser);

 // At this time we are switching to warning message and messages
 // which are going to follow it are going to be warnings message too.
 // These messages can be displayed only if the user level (sets with
 // linear_algebra::setlevel function) is set to linear_algebra::warnings.
 linear_algebra::log << linear_algebra::warnings;

 // With the following linear_algebra::file function we are defining that
 // all future logs are going to this new file resource which is
 // test.txt
 linear_algebra::log << linear_algebra::file("test.txt") << "In FILE" << std::endl;

 // Now we are changing again the resources destination to cout which
 // is the standard output.
 linear_algebra::log << std::cout << "In COUT" << std::endl;

 // Here are 2 differents examples of how to set the errors user level
 // in using either a string or an identifier.
 linear_algebra::log << linear_algebra::setlevel("errors");
 linear_algebra::log << linear_algebra::setlevel(linear_algebra::errors);

 // Now we are writting a message, that will be displayed only if we are above the "quiet" level
 linear_algebra::log << linear_algebra::quiet << "1) Must be in quiet mode to see that" << std::endl;

 // And so on...
 linear_algebra::log << linear_algebra::setlevel(linear_algebra::warnings) << linear_algebra::warnings << "2) Must be in warnings mode to see that" << std::endl;

 linear_algebra::log << linear_algebra::setlevel(linear_algebra::logging);

 linear_algebra::log << linear_algebra::errors;
 linear_algebra::log << "3) Must be in errors mode to see that";
 linear_algebra::log << std::endl;

 linear_algebra::log << linear_algebra::debug << 4 << ')'
 << " Must be in debug mode to see that\n";

 return 0;
 }
 \endcode

 @{
*/

#ifndef linear_algebra_Logger_h
#define linear_algebra_Logger_h

#include <map>
#include <vector>
#include <string>
#include <iosfwd>

#include "Object.h"

namespace linear_algebra
{
    /**
     * Levels contents all the available levels in Logger
     *
     * /!\ If you want to add a level dont forget to add it at the implementation of the class Logger in the ctor
     */
    enum Levels {quiet = 0,
		 errors,
		 warnings,
		 progress,
		 logging,
		 debug,
		 xdebug};

    /**
     * file
     * this structure combined with the friend operator<< below is an easy way to select a file as output.
     */
    struct	file
    {
	file(const std::string f);
	const std::string _f;
    };

    /**
     * setlevel
     * this structure combined with the friend operator<< below is an easy way to set a verbose level.
     */
    struct	setlevel
    {
	setlevel(const std::string v);
	setlevel(const Levels lvl);
	const std::string _v;
	const Levels _lvl;
    };

    /**
     * Logger
     * Class providing a verbose management through EO
     * Use of a global variable linear_algebra::log to easily use the logger like std::cout
     */
    class	Logger : public Object,
			   public std::ostream
    {
    public:
	Logger();
	~Logger();

	virtual std::string	className() const;

	//! Print the available levels on the standard output
	void printLevels() const;

	/*! Returns the selected levels, that is the one asked by the user
	 *
	 * Use this function if you want to be able to compare selected levels to a given one, like:
	 * if( linear_algebra::log.getLevelSelected() >= linear_algebra::progress ) {...}
	 */
	Levels getLevelSelected() const { return _selectedLevel; }

	/*! Returns the current level of the context
	 * the one given when you output message with the logger
	 */
	Levels getLevelContext() const { return _contextLevel; }

    protected:
	void	addLevel(std::string name, Levels level);

    private:
	/**
	 * outbuf
	 * this class inherits from std::streambuf which is used by Logger to write the buffer in an output stream
	 */
	class	outbuf : public std::streambuf
	{
	public:
	    outbuf(const int& fd,
		   const Levels& contexlvl,
		   const Levels& selectedlvl);
	protected:
	    virtual int	overflow(int_type c);
	private:
	    const int&		_fd;
	    const Levels&	_contextLevel;
	    const Levels&	_selectedLevel;
	};

    private:
	/**
	 * MapLevel is the type used by the map member _levels.
	 */
	typedef std::map<std::string, Levels>	MapLevel;

    public:
	/**
	 * operator<< used there to set a verbose mode.
	 */
	friend Logger&	operator<<(Logger&, const Levels);

	/**
	 * operator<< used there to set a filename through the class file.
	 */
	friend Logger&	operator<<(Logger&, file);

	/**
	 * operator<< used there to set a verbose level through the class setlevel.
	 */
	friend Logger&	operator<<(Logger&, setlevel);

	/**
	 * operator<< used there to be able to use std::cout to say that we wish to redirect back the buffer to a standard output.
	 */
	friend Logger&	operator<<(Logger&, std::ostream&);

    private:
	/**
	 * _selectedLevel is the member storing verbose level setted by the user thanks to operator()
	 */
	Levels	_selectedLevel;
	Levels	_contextLevel;

	/**
	 * _fd in storing the file descriptor at this place we can disable easily the buffer in
	 * changing the value at -1. It is used by operator <<.
	 */
	int		_fd;

	/**
	 * _obuf std::ostream mandates to use a buffer. _obuf is a outbuf inheriting of std::streambuf.
	 */
	outbuf	_obuf;

	/**
	 * _levels contains all the existing level order by position
	 */
	MapLevel	_levels;

	/**
	 * _levelsOrder is just a list to keep the order of levels
	 */
	std::vector<std::string>	_sortedLevels;

	/**
	 * _standard_io_streams
	 */
	std::map< std::ostream*, int >	_standard_io_streams;
    };
    /** @example t-Logger.cpp
     */

    /**
     * log is an external global variable defined to easily use a same way than std::cout to write a log.
     */
    extern Logger	log;

}

/** @} */

#endif // !linear_algebra_Logger_h
