#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
def mark_output(ys):
    if type(ys) != tuple:
        ys = (ys, )
    for i in range(len(ys)):
        y = ys[i]
        if type(y) == tuple:
            for yy in y:
                y_shape = [d._attrs["values"] for d in yy._attrs["shape"]]
                y_name = yy._attrs["name"]
                print("AIT {} shape: {}".format(y_name, y_shape))
        else:
            y_shape = [d._attrs["values"] for d in y._attrs["shape"]]
            y_name = y._attrs["name"]
            print("AIT {} shape: {}".format(y_name, y_shape))
