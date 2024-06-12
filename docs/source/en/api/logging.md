<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Logging

🤗 Diffusers has a centralized logging system to easily manage the verbosity of the library. The default verbosity is set to `WARNING`.

To change the verbosity level, use one of the direct setters. For instance, to change the verbosity to the `INFO` level.

```python
import diffusers

diffusers.logging.set_verbosity_info()
```

You can also use the environment variable `DIFFUSERS_VERBOSITY` to override the default verbosity. You can set it
to one of the following: `debug`, `info`, `warning`, `error`, `critical`. For example:

```bash
DIFFUSERS_VERBOSITY=error ./myprogram.py
```

Additionally, some `warnings` can be disabled by setting the environment variable
`DIFFUSERS_NO_ADVISORY_WARNINGS` to a true value, like `1`. This disables any warning logged by
[`logger.warning_advice`]. For example:

```bash
DIFFUSERS_NO_ADVISORY_WARNINGS=1 ./myprogram.py
```

Here is an example of how to use the same logger as the library in your own module or script:

```python
from diffusers.utils import logging

logging.set_verbosity_info()
logger = logging.get_logger("diffusers")
logger.info("INFO")
logger.warning("WARN")
```


All methods of the logging module are documented below. The main methods are
[`logging.get_verbosity`] to get the current level of verbosity in the logger and
[`logging.set_verbosity`] to set the verbosity to the level of your choice.

In order from the least verbose to the most verbose:

|                                                    Method | Integer value |                                         Description |
|----------------------------------------------------------:|--------------:|----------------------------------------------------:|
| `diffusers.logging.CRITICAL` or `diffusers.logging.FATAL` |            50 |                only report the most critical errors |
|                                 `diffusers.logging.ERROR` |            40 |                                  only report errors |
|   `diffusers.logging.WARNING` or `diffusers.logging.WARN` |            30 |           only report errors and warnings (default) |
|                                  `diffusers.logging.INFO` |            20 | only report errors, warnings, and basic information |
|                                 `diffusers.logging.DEBUG` |            10 |                              report all information |

By default, `tqdm` progress bars are displayed during model download. [`logging.disable_progress_bar`] and [`logging.enable_progress_bar`] are used to enable or disable this behavior.

## Base setters

[[autodoc]] utils.logging.set_verbosity_error

[[autodoc]] utils.logging.set_verbosity_warning

[[autodoc]] utils.logging.set_verbosity_info

[[autodoc]] utils.logging.set_verbosity_debug

## Other functions

[[autodoc]] utils.logging.get_verbosity

[[autodoc]] utils.logging.set_verbosity

[[autodoc]] utils.logging.get_logger

[[autodoc]] utils.logging.enable_default_handler

[[autodoc]] utils.logging.disable_default_handler

[[autodoc]] utils.logging.enable_explicit_format

[[autodoc]] utils.logging.reset_format

[[autodoc]] utils.logging.enable_progress_bar

[[autodoc]] utils.logging.disable_progress_bar
