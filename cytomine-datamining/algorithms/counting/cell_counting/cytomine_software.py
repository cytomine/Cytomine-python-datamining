# -*- coding: utf-8 -*-

# * Copyright (c) 2009-2017. Authors: see NOTICE file.
# *
# * Licensed under the Apache License, Version 2.0 (the "License");
# * you may not use this file except in compliance with the License.
# * You may obtain a copy of the License at
# *
# *      http://www.apache.org/licenses/LICENSE-2.0
# *
# * Unless required by applicable law or agreed to in writing, software
# * distributed under the License is distributed on an "AS IS" BASIS,
# * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# * See the License for the specific language governing permissions and
# * limitations under the License.


__author__ = "Rubens Ulysse <urubens@uliege.be>"
__copyright__ = "Copyright 2010-2017 University of Liège, Belgium, http://www.cytomine.be/"

# TODO: transfer to cytomine_utilities


class InstallParameter(object):
    """
    A helper class to wrap software parameter properties during installation.
    """

    def __init__(self, name, type_, default_value, index, required=True, set_by_server=False,
                 uri=None, uri_print_attr=None, uri_sort_attr=None):
        """
        Initialize a wrapper class for software parameter during installation.

        Parameters
        ----------
        name: str
            The parameter name
        type_: type or str
            The type of parameter (int, float, bool, str, "Number", "Boolean", "String", "Domain", "ListDomain")
        default_value: str
            The default value
        index: int
            The parameter index in the list of parameters
        required: bool (default=True)
            If required for every execution
        set_by_server: bool (default=False)
            If the value is automatically set by the server
        uri: str
        uri_print_attr: str
        uri_sort_attr: str
        """
        self.name = name
        self.type = self._type2str(type_)
        self.default_value = default_value
        self.index = index
        self.required = required
        self.set_by_server = set_by_server
        self.uri = uri
        self.uri_print_attr = uri_print_attr
        self.uri_sort_attr = uri_sort_attr

    def _type2str(self, type_):
        if type_ == str:
            return "String"
        elif type_ == int or type_ == float:
            return "Number"
        elif type_ == bool:
            return "Boolean"
        else:
            return type_

    def add_software_parameter(self, cytomine, software_id):
        cytomine.add_software_parameter(self.name, software_id, self.type, self.default_value,
                                        self.required, self.index, self.set_by_server, uri=self.uri,
                                        uriPrintAttribut=self.uri_print_attr, uriSortAttribut=self.uri_sort_attr)

    def parameter_command(self):
        return "--{} ${} ".format(self.name, self.name)


class InstallSoftware(object):
    """
    A helper class to wrap software properties during installation.
    """

    def __init__(self, name, service_name, result_name, software_router=False, software_path=None,
                 software_working_path=None):
        """
        Initialize an helper class to wrap software properties during installation.

        Parameters
        ----------
        name: str
            The software name for Cytomine
        service_name: str
            The Cytomine service name
            - pyxitSuggestedTermJobService for CLI execution (should be improved)
            - createRabbitJobWithArgsService for Web-UI execution of Python scripts with ArgumentParser
        result_name: str
            The result type
            - ValidateAnnotation
            - Default
        software_router: bool, default=False
            True if the software needs to be installed on the software router.
            In this case, the software_path must be completed and the service_name
            is automatically fixed to "createRabbitJobWithArgsService".
        software_path: str, default=None
            The path where are stored the algorithms in the software router
        software_working_path: str, default=None
            The path where working data are stored in the software router
        """
        self.name = name
        self.service_name = service_name if not software_router else "createRabbitJobWithArgsService"
        self.result_name = result_name
        self.software_path = software_path
        self.software_working_path = software_working_path
        self.software_router = software_router
        self.parameters = list()
        self.index = 0

    def add_parameter(self, name, type_, default_value, required=True, set_by_server=False,
                      uri=None, uri_print_attr=None, uri_sort_attr=None):
        self.index += 100
        self.parameters.append(InstallParameter(name, type_, default_value, self.index, required,
                                                set_by_server, uri, uri_print_attr, uri_sort_attr))

    def install_software(self, cytomine):
        command = None
        if self.software_router:
            command = self._make_command()

        software = cytomine.add_software(self.name, self.service_name, self.result_name, command)
        for parameter in self.parameters:
            parameter.add_software_parameter(cytomine, software.id)

        return software

    def _make_command(self):
        command = "python {} ".format(self.software_path)
        command += "--cytomine_host $host "
        command += "--cytomine_public_key $publicKey "
        command += "--cytomine_private_key $privateKey "
        command += "--cytomine_base_path /api/ "
        command += "--cytomine_working_path {} ".format(self.software_working_path)
        for parameter in self.parameters:
            command += parameter.parameter_command()

        return command
