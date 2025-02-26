# Copyright 2025 The HuggingFace Team. All rights reserved.
#
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
"""
Typing utilities: Utilities related to type checking and validation
"""

from typing import Any, Dict, List, Set, Tuple, Type, Union, get_args, get_origin


def _is_valid_type(obj: Any, class_or_tuple: Union[Type, Tuple[Type, ...]]) -> bool:
    """
    Checks if an object is an instance of any of the provided types. For collections, it checks if every element is of
    the correct type as well.
    """
    if not isinstance(class_or_tuple, tuple):
        class_or_tuple = (class_or_tuple,)

    # Unpack unions
    unpacked_class_or_tuple = []
    for t in class_or_tuple:
        if get_origin(t) is Union:
            unpacked_class_or_tuple.extend(get_args(t))
        else:
            unpacked_class_or_tuple.append(t)
    class_or_tuple = tuple(unpacked_class_or_tuple)

    if Any in class_or_tuple:
        return True

    obj_type = type(obj)
    # Classes with obj's type
    class_or_tuple = {t for t in class_or_tuple if isinstance(obj, get_origin(t) or t)}

    # Singular types (e.g. int, ControlNet, ...)
    # Untyped collections (e.g. List, but not List[int])
    elem_class_or_tuple = {get_args(t) for t in class_or_tuple}
    if () in elem_class_or_tuple:
        return True
    # Typed lists or sets
    elif obj_type in (list, set):
        return any(all(_is_valid_type(x, t) for x in obj) for t in elem_class_or_tuple)
    # Typed tuples
    elif obj_type is tuple:
        return any(
            # Tuples with any length and single type (e.g. Tuple[int, ...])
            (len(t) == 2 and t[-1] is Ellipsis and all(_is_valid_type(x, t[0]) for x in obj))
            or
            # Tuples with fixed length and any types (e.g. Tuple[int, str])
            (len(obj) == len(t) and all(_is_valid_type(x, tt) for x, tt in zip(obj, t)))
            for t in elem_class_or_tuple
        )
    # Typed dicts
    elif obj_type is dict:
        return any(
            all(_is_valid_type(k, kt) and _is_valid_type(v, vt) for k, v in obj.items())
            for kt, vt in elem_class_or_tuple
        )

    else:
        return False


def _get_detailed_type(obj: Any) -> Type:
    """
    Gets a detailed type for an object, including nested types for collections.
    """
    obj_type = type(obj)

    if obj_type in (list, set):
        obj_origin_type = List if obj_type is list else Set
        elems_type = Union[tuple({_get_detailed_type(x) for x in obj})]
        return obj_origin_type[elems_type]
    elif obj_type is tuple:
        return Tuple[tuple(_get_detailed_type(x) for x in obj)]
    elif obj_type is dict:
        keys_type = Union[tuple({_get_detailed_type(k) for k in obj.keys()})]
        values_type = Union[tuple({_get_detailed_type(k) for k in obj.values()})]
        return Dict[keys_type, values_type]
    else:
        return obj_type
