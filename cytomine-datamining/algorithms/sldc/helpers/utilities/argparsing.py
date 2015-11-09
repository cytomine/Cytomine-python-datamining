# -*- coding: utf-8 -*-
"""
Copyright 2010-2013 University of Liège, Belgium.

This software is provided 'as-is', without any express or implied warranty.
In no event will the authors be held liable for any damages arising from the
use of this software.

Permission is only granted to use this software for non-commercial purposes.
"""

__author__ = "Begon Jean-Michel <jm.begon@gmail.com>"
__copyright__ = "Copyright 2010-2013 University of Liège, Belgium"
__version__ = '0.1'


import logging
import argparse
from cytomine import Cytomine

def positive_int(string):
    """
    Check whether a given string is a positive integer
    """
    val = int(string)
    if val < 0:
        raise argparse.ArgumentTypeError("%s is not a positive integer value" % string)
    return val
    # strings = string.split(" ")
    # vals = []
    # for s in strings:
    #     val = int(s)
    #     if val < 0:
    #         raise argparse.ArgumentTypeError("%s is not a positive integer value" % string)
    #     vals.append(val)
    # return vals

def strictly_positive_int(string):
    """
    Check whether a given string is a strictly positive integer
    """
    val = int(string)
    if val <= 0:
        raise argparse.ArgumentTypeError("%s is not a strictly positive integer value" % string)
    return val

def range0_255(string):
    """
    Check whether a given string is a positive integer in the range [0, 255]
    """
    val = int(string)
    if val < 0 or val > 255:
        raise argparse.ArgumentTypeError("%s is not a positive integer value in the range [0, 255]" % string)
    return val

def positive_float(string):
    """
    Check whether a given string is a positive float
    """
    val = float(string)
    if val < 0:
        raise argparse.ArgumentTypeError("%s is not a positive float value" % string)
    return val

def not_zero(string):
    """
    Check wheter a given string is a non-zero integer
    """
    val = int(string)
    if val == 0:
        raise argparse.ArgumentTypeError("requiring non-zero integer")
    return val


def under_unit(string):
    """
    Check wheter a given string is a float smaller or equal to one
    """
    val = float(string)
    if val > 1:
        raise argparse.ArgumentTypeError("%s is greater than one" % string)
    return val

def check_verbosity(string):
    """
    Check the verbosity
    """
    if string == "debug":
        return string
    if string == "info":
        return string
    if string == "warn":
        return string
    if string == "error":
        return string
    if string == "critical":
        return string
    raise argparse.ArgumentTypeError("%s is not one of {'debug', 'info', 'warn', 'error', 'critical'}" % string)

def fillin_cytomine_args(parser):
    """
    Fill in the :class:`ArgumentParser` instance with the Cytomine
    command line positional and optional arguments.

    Beware of the position in which you insert this lot

    Cytomine arguments
    ------------------
    host : str
        Cytomine server host URL
    public_key : str
        User public key
    private_key : str
        User Private key
    software_id : str
        Identifier of the software on the Cytomine server
    project_id : str
        Identifier of the project to process on the Cytomine
    --working_path : path (default : '/tmp')
        Directory for caching temporary files
    --protocol : str (default : 'http://')
        Communication protocol
    --base_path : str (default : '/api/')
        n/a
    --timeout : int >= 0 (default : 120)
        Timeout time for connection (in seconds)

    Parameters
    ----------
    parser : :class:`ArgumentParser`
        The instance to fill in
    """
    parser.add_argument("host",
                        help="Cytomine server host URL")
    parser.add_argument("public_key",
                        help="User public key")
    parser.add_argument("private_key",
                        help="User Private key")
    parser.add_argument("software_id",
                        help="Identifier of the software on the Cytomine server")
    parser.add_argument("project_id",
                        help="Identifier of the project to process on the Cytomine server")
    parser.add_argument("--working_path",
                        help="Directory for caching temporary files",
                        default="/tmp")
    parser.add_argument("--protocol",
                        help="Communication protocol",
                        default="http://")
    parser.add_argument("--base_path",
                        help="n/a",
                        default="/api/")
    parser.add_argument("--timeout",
                        help="Timeout time for connection (in seconds)",
                        type=positive_int,
                        default="120")

def cytomine_from_argparse(parser, verbose=None):
    """
    Transform the given namespace instance into
    a :class:`Cytomine` instance.

    The parser must have been filled in by the proc:`fillin_cytomine_args`

    Parameters
    ----------
    parser : namesapce
        The namesapce containing the arguments

    Return
    ------
    cytomine : class:`Cytomine`
        The newly created Cytomine instance
    """
    namesapce = parser.parse_args()
    if verbose is None:
        if hasattr(namesapce, "verbose"):
            verbose = namesapce.verbose
        else:
            verbose = False

    return Cytomine(namesapce.host, namesapce.public_key,
                    namesapce.private_key, namesapce.working_path,
                    namesapce.protocol, namesapce.base_path,
                    verbose, namesapce.timeout)



def print_args(args, excepts=None):
    """
    Prints the arguments

    Parameters
    ----------
    args : namespace returned by argparse
        The arguments to print
    excepts : container of string or None (default : None)
        The strings not to print
    """
    for key, val in vars(args).items():
        if (excepts is None) or (not key in excepts):
            print "%s : %s" % (str(key), str(val))

class VerbosityAction(argparse.Action):
    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        if nargs is not None:
            raise ValueError("nargs not allowed")
        argparse.Action.__init__(self, option_strings, dest, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        if values == "debug":
            setattr(namespace, self.dest, logging.DEBUG)
        elif values == "info":
            setattr(namespace, self.dest, logging.INFO)
        elif values == "warn":
            setattr(namespace, self.dest, logging.WARN)
        elif values == "error":
            setattr(namespace, self.dest, logging.ERROR)
        elif values == "critical":
            setattr(namespace, self.dest, logging.CRITICAL)
