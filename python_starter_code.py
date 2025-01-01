#!/usr/bin/env python3.11

# By Ran Novitsky Nof (ran.nof@gmail.com), 2024 @ GSI
# Update Jul 29, 2024 R.N.N @ GSI

# import modules
import argparse
import logging
from logging.handlers import TimedRotatingFileHandler


# Auxilary parmeters for logging
_LOG_LEVEL_STRINGS = ['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG']
formatter = logging.Formatter(fmt='%(asctime)s.%(msecs)03d | %(name)s | %(levelname)s | %(message)s',
                              datefmt='%Y-%m-%dT%H:%M:%S')


# Default parameers for logging
VERBOSE = False  # Turn to True for printing to terminal
LOG_FILE = None  # Set a file name for logging into a file 
LOG_LEVEL = 'CRITICAL'  # Change to any of the _LOG_LEVEL_STRINGS values. See documentation of built-in logging module
log = logging.getLogger('PROJECTNAME')  # Replace with the name of the project or application
log.setLevel(LOG_LEVEL)  # set at default log level

# default parameter for the code. Can be override by a config file (-c commandline options). 
# Simply copy the lines for override into the config file and edit the values.
example_parameter1 = 'example_value1'
example_parameter2 = 'example_value2'
example_parameter3 = 'example_value3'


DESCRIPTION="""
This is the description of the project, you can add any relevant information.
"""

# Setting commandline argument using a parser
parser = argparse.ArgumentParser(
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description=DESCRIPTION,
    epilog='''Created by Ran Novitsky Nof (ran.nof@gmail.com), 2024 @ GSI''')
# argumants for logging
parser.add_argument('-v', help='verbose - print messages to screen?', action='store_true', default=VERBOSE)
parser.add_argument('-l', '--log_level', choices=_LOG_LEVEL_STRINGS, default=LOG_LEVEL,
                    help=f"Log level (Default: {LOG_LEVEL}). see Python's Logging module for more details")
parser.add_argument('--logfile', metavar='log file name', help='log to file', default=LOG_FILE)
# arguments for a config file
parser.add_argument('-c','--configfile', default=None, help='Config file.')
# ADD ANY OTHER ARGUMENTS HERE
parser.add_argument('input_parameter', help='Description of the input parameter')


def set_logger(log, verbose=VERBOSE, log_level=LOG_LEVEL, logfile=LOG_FILE):
    """A function for setting the logger.
    log: The logger needed to be set. Typically set this log at the top of the file with logging.get(LOGGERNAME).
    verbose: True (print to screen) or False (do not print to screen)
    log_level: A valid logging level ('CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG')
    logfile: A file name (including path). The file is a daily file and for long runs will change at midnight. See logging documentation for TimedRotatingFileHandler
    """

    log.setLevel(log_level)
    if verbose:
        # create console handler
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(log_level)
        ch.setFormatter(formatter)
        if logging.StreamHandler not in [h.__class__ for h in log.handlers]:
            log.addHandler(ch)
        else:
            log.warning('log Stream handler already applied.')
    if logfile:
        # create file handler
        fh = TimedRotatingFileHandler(logfile,
                                  when='midnight',
                                  utc=True)
        fh.setLevel(log_level)
        fh.setFormatter(formatter)
        if TimedRotatingFileHandler not in [h.__class__ for h in log.handlers]:
            log.addHandler(fh)
        else:
            log.warning('Log file handler already applied.')
        log.info(f'Log file is: {logfile}')
    else:
        log.debug('No Log file was set')

"""
# REST OF THE CLASSES AND FUNCTIONS HERE


# will run only when using this file as an exacutable
if __name__=='__main__':
    # get user command line values
    args = parser.parse_args()
    # set the logger
    set_logger(log, args.v, args.log_level, args.logfile)
    # update default parameters
    if args.configfile:
        try:
            with open(args.configfile, 'r') as f:
                for line in f:
                    exec(line)
        except Exception as err:
            parser.error(f"Error reading config file {args.configfile}:\n{err}.")
    # MORE CODE HERE
    """





