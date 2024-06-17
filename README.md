# polus-classic-segmentation

[![License BSD-3](https://img.shields.io/pypi/l/polus-classic-segmentation.svg?color=green)](https://github.com/feldmangn/polus-classic-segmentation/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/polus-classic-segmentation.svg?color=green)](https://pypi.org/project/polus-classic-segmentation)
[![Python Version](https://img.shields.io/pypi/pyversions/polus-classic-segmentation.svg?color=green)](https://python.org)
[![tests](https://github.com/feldmangn/polus-classic-segmentation/workflows/tests/badge.svg)](https://github.com/feldmangn/polus-classic-segmentation/actions)
[![codecov](https://codecov.io/gh/feldmangn/polus-classic-segmentation/branch/main/graph/badge.svg)](https://codecov.io/gh/feldmangn/polus-classic-segmentation)
[![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/polus-classic-segmentation)](https://napari-hub.org/plugins/polus-classic-segmentation)

Classic sergmentation module from Polus
The following models from PolusAI are featured in this plugin:

Polus Cell Nuclei Segmentation
Original Code used for WIPP Plugin "Cell Nuclei Segmentation using U-net"
Their model features the neural network and model weights from https://github.com/axium/Data-Science-Bowl-2018/. This original model has been modified to use reflective padding to make the dimension a multiple of 256. Then, the loop extracts 256x256 tiles to be processed by the network. Then, it untiles and removes the padding from the output.

Curvy Linear
Workflow for curvilinear shapes such as: Sec61 beta, TOM20, lamin B1 (mitosis specific)

Spotty
Workflow for data with a spotty appearance in each 2d frame such as fibrillarin and beta catenin.


----------------------------------

This [napari] plugin was generated with [Cookiecutter] using [@napari]'s [cookiecutter-napari-plugin] template.

<!--
Don't miss the full getting started guide to set up your new package:
https://github.com/napari/cookiecutter-napari-plugin#getting-started

and review the napari docs for plugin developers:
https://napari.org/stable/plugins/index.html
-->

## Installation

You can install `polus-classic-segmentation` via [pip]:

    pip install polus-classic-segmentation



To install latest development version :

    pip install git+https://github.com/feldmangn/polus-classic-segmentation.git


## Contributing

Contributions are very welcome. Tests can be run with [tox], please ensure
the coverage at least stays the same before you submit a pull request.

## License

Distributed under the terms of the [BSD-3] license,
"polus-classic-segmentation" is free and open source software

## Issues

If you encounter any problems, please [file an issue] along with a detailed description.

[napari]: https://github.com/napari/napari
[Cookiecutter]: https://github.com/audreyr/cookiecutter
[@napari]: https://github.com/napari
[MIT]: http://opensource.org/licenses/MIT
[BSD-3]: http://opensource.org/licenses/BSD-3-Clause
[GNU GPL v3.0]: http://www.gnu.org/licenses/gpl-3.0.txt
[GNU LGPL v3.0]: http://www.gnu.org/licenses/lgpl-3.0.txt
[Apache Software License 2.0]: http://www.apache.org/licenses/LICENSE-2.0
[Mozilla Public License 2.0]: https://www.mozilla.org/media/MPL/2.0/index.txt
[cookiecutter-napari-plugin]: https://github.com/napari/cookiecutter-napari-plugin

[file an issue]: https://github.com/feldmangn/polus-classic-segmentation/issues

[napari]: https://github.com/napari/napari
[tox]: https://tox.readthedocs.io/en/latest/
[pip]: https://pypi.org/project/pip/
[PyPI]: https://pypi.org/
