# -*- coding: utf-8 -*-
"""
Copyright 2010-2013 University of LiÃ¨ge, Belgium.

This software is provided 'as-is', without any express or implied warranty.
In no event will the authors be held liable for any damages arising from the
use of this software.

Permission is only granted to use this software for non-commercial purposes.
"""

__author__ = "Begon Jean-Michel <jm.begon@gmail.com>"
__copyright__ = "Copyright 2010-2013 University of LiÃ¨ge, Belgium"
__version__ = '0.1'

from cytomine.cytomine import Cytomine

#TODO progress callback/logger to inform Cytomine


def create_cytomine_job(host,
                        public_key,
                        private_key,
                        software_id,
                        project_id,
                        working_path="/tmp",
                        protocol="http://",
                        base_path="/api/",
                        verbose=False,
                        timeout=120):
    """
    create a :class:`CytomineJob`

    Parameters
    ----------
    host : str
        The Cytomine server host URL
    public_key : str
        The user public key
    private_key : str
        The user corresponding private key
    software_id : int
        The identifier of the software on the Cytomine server
    project_id : int
        The identifier of the project to process on the Cytomine server
    slide_ids : sequence of int
        The identifiers of the slides to process
    zoom_sl : int >= 0 (default : 0)
        The zoom level for the segment-locate part
    working_path : str (default : "/tmp")
        A directory for caching temporary files
    protocol : str (default : "http://")
        The communication protocol
    base_path : str (default : /api/)
        n/a
    verbose : boolean (default : False)
        Whether to display messages or not
    timeout : int (default : 120)
        The timeout time (in seconds)
    """
    cytomine_client = Cytomine(host, public_key, private_key, working_path,
                               protocol, base_path, verbose, timeout)
    return CytomineJob(cytomine_client, software_id, project_id)


class CytomineJob:
    """
    ===========
    CytomineJob
    ===========
    A :class:`CytomineJob` represents a job in the cytomine context.
    This class does nothing by itself. It is just supposed to be
    inherited from or incorporated in any class representing
    a software/job registered in the Cytomine server

    Usage
    -----
    Either use the :meth:`connect`/meth:`close`methods or use it with a 
    with statement:
     
    with CytomineJob(...) as job:
         do_your_stuff()

    Constructor parameters
    ----------------------
    cytomine_client : :class:`Cytomine`
        The Cytomine client through which to communicate with the server
    software_id : int
        The identifier of the software on the Cytomine server
    project_id : int
        The identifier of the project to process on the Cytomine server
    """
    def __init__(self, cytomine_client, software_id, project_id):
        self.__cytomine = cytomine_client
        self.__software_id = software_id
        self.__project_id = project_id
        self.__job_done = False
        self.__job = None

    @property
    def cytomine_client(self):
        """
        Protected method

        Return
        ------
        cytomine : :class:`Cytomine`
            The Cytomine client
        """
        return self.__cytomine

    @property
    def project_id(self):
        """
        Protected method

        Return
        ------
        project_id : int
            The id of the project
        """
        return self.__project_id

    @property
    def software_id(self):
        """
        Protected method

        Return
        ------
        software_id : int
            The id of the software
        """
        return self.__software_id

    def done(self, status=True):
        """
        Indicates whether the job is finished or not

        Parameters
        ----------
        status : bool
            Whether the process is finished
        """
        self.__job_done = status

    def is_done(self):
        """
        Return
        ------
        job_status : bool
            Whether the process is finished
        """
        return self.__job_done

    def start(self):
        """
        Connect to the Cytomine server and switch to job connection
        Incurs dataflows
        """
        user_job = self.__cytomine.add_user_job(self.__software_id,
                                                self.__project_id)
        self.__cytomine.set_credentials(str(user_job.publicKey),
                                        str(user_job.privateKey))
        
        job = self.__cytomine.get_job(user_job.job)
        self.__job = self.__cytomine.update_job_status(job, status=job.RUNNING)  # Dataflow

    def close(self):
        """
        Notify the Cytomine server of the job's end
        Incurs a dataflow
        """
        status = self.__job.FAILED
        if self.is_done():
            status = self.__job.TERMINATED

        self.__cytomine.update_job_status(self.__job, status=status)  # Dataflow

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, type, value, traceback):
        if value is None:
            # No exception, job is done
            self.done()
        self.close()
        return False
