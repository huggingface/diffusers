# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Utilities for constructing PyTrees of PartitionSpecs."""

# utils adapted from https://github.com/google-research/google-research/blob/master/flax_models/t5x/partitions.py

from jax.sharding import PartitionSpec as P 
from jax.sharding import NamedSharding

# For specifying empty leaf dict `{}`
def partition_shape(shape):
  if len(shape) == 1:
    if shape[0] % 4 == 0:
      return P("dp")
    elif shape[0] % 2 == 0:
      return P("mp")
  if len(shape) == 2:
    if shape[0] % 4 == 0 and shape[1] % 2 == 0 and shape[0] > shape[1]:
      return P("dp","mp")
    if shape[0] % 2 == 0 and shape[1] % 4 == 0:
      return P("mp","dp")
    if shape[0] % 4 == 0:
      return P("dp",None)
    if shape[1] % 4 == 0:
      return P(None,"dp")
    if shape[0] % 2 == 0 and shape[1] % 2 == 0:
      return P("mp",None)
  if len(shape) == 4:
    if shape[-2] % 4 == 0 and shape[-1] % 2 == 0:
      return P(None,None,"dp","mp")
    if shape[-2] % 2 == 0 and shape[-1] % 4 == 0:
      return P(None,None,"mp","dp")
    if shape[-2] % 4 == 0:
      return P(None,None,"dp",None)
    if shape[-1] % 4 == 0:
      return P(None,None,None,"dp")
    if shape[-1] % 2 == 0 and shape[-2] % 2 == 0:
      return P(None,None,"mp",None)
    
  return P()
