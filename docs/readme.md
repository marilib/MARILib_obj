# Doc generation with Sphinx

Initializing docs
=================
This folder docs/ as been created once for all using the command
    
    MARILib_obj/docs/> sphinx-quickstart

Building the html documentation
===============================

Then to build the documentation, run the three following commands.
First clean the `build/` directory with

    MARILib_obj/docs/> make clean
    
Then use Apidoc to auto-generate the documentation from the docstring in all packages, subpackages and modules:

    MARILib_obj/docs/> sphinx-apidoc -fMe ../marilib/ -o source/api/ -t source/_templates/apidoc/

The different options are (see [sphinx-apidoc](https://www.sphinx-doc.org/en/master/man/sphinx-apidoc.html) for more details):

* -f : Force overwriting of any existing generated files
* -M : Put module documentation before submodule documentation
* -e : Put documentation for each module on its own page
* -t : use the specified templates to generate the docs

and finally build the html pages:

    MARILib_obj/docs/> make html
    