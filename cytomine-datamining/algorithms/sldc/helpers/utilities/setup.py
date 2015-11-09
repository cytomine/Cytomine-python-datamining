# -*- coding: utf-8 -*-

#from distutils.core import setup
#
#setup(
#    name='Cytomine Python Utilities',
#    version='0.1',
#    author='Benjamin Stévens',
#    author_email='b.stevens@ulg.ac.be',
#    packages=['adapter',
#              'datatype',
#              'misc',
#              'source'],
#    url='http://www.cytomine.be',
#    license='LICENSE.txt',
#    description='Cytomine Python Client.',
#    long_description=open('README.txt').read(),
#    install_requires=[
#        "Cytomine-Python-Client >= 0.1",
#        "numpy",
#        "shapely",
#        "pil"
#        #"pyopencv"
#    ],
#    dependency_links = [
#        'git+ssh://git@github.com/cytomine/Cytomine-Python-Client.git#egg=Cytomine-Python-Client-0.1'
#    ]
#)


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration('cytomine_utilities', parent_package, top_path)
    config.add_subpackage('datatype')
    config.add_subpackage('source')

    return config
