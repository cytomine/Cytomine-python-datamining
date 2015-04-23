# -*- coding: utf-8 -*-


#
# * Copyright (c) 2009-2015. Authors: see NOTICE file.
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
# */

__author__          = "Gilles Louppe"
__contributors__    = ["Marée Raphaël <raphael.maree@ulg.ac.be>", "Stévens Benjamin <b.stevens@ulg.ac.be>"]
__copyright__       = "Copyright 2010-2015 University of Liège, Belgium, http://www.cytomine.be/"


import numpy as np
import os
import os.path
try:
    import Image
except:
    from PIL import Image



from sklearn.utils import check_random_state

def build_from_dir(directory, map_classes = None):
    X = []
    y = []

    for c in os.listdir(directory):
        for _file in os.listdir(os.path.join(directory, c)):
            try:
                image = Image.open(os.path.join(directory, c, _file))
                image.verify()
                X.append(os.path.join(directory, c, _file))
                if map_classes:
                    y.append(map_classes[int(c)])
                else:
                    y.append(c)
            except IOError:
                print "warning filename %s is not an image" % os.path.join(directory, c, _file)

    X = np.array(X)
    y = np.array(y)

    return X, y
