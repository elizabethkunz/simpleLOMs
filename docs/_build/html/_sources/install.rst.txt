Installation
============

Install from PyPI
-----------------

The recommended way to install **simpleLOMs** is from PyPI using pip:

.. code-block:: bash

   pip install simpleLOMs

Install from source
-------------------

To install the latest development version from the repository:

.. code-block:: bash

   git clone https://github.com/elizabethkunz/simpleLOMs.git
   cd simpleLOMs
   pip install -e .

For a non-editable install, omit the ``-e`` flag.

Building the documentation locally
----------------------------------

From the repository root:

.. code-block:: bash

   python3 -m pip install -e ".[docs]"
   cd docs
   make cleanhtml
   open _build/html/index.html

Use ``make cleanhtml`` (not plain ``make html``) whenever you add, remove, or
rename pages in the toctree — otherwise older pages can keep a stale sidebar.

The ``.[docs]`` install pulls in **simpleLOMs** and its runtime dependencies
(``numpy``, ``scipy``, ``scikit-rf``, …) plus Sphinx / Furo / nbsphinx.
Autodoc needs those runtime packages present; without them the API reference
only documents modules that import with no scientific stack (today, just
``CPWParams``).

Notebooks are rendered without execution (see ``nbsphinx_execute`` in
``conf.py``), so cells are not re-run during the doc build.