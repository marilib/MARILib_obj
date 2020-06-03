"""The aircraft package is the main package of MARILib.

Usage
=====

To initialize an airplane you need to import the package :mod:`marilib.aircraft` and give it a name::

    import marilib.aircraft

    myplane = Aircraft("my Plane")

The statement `import marilib.aircraft` will load :class:`aircraft_root.Aircraft`, :class:`aircraft_root.Arrangement`
and :class:`requirement.Requirement`. At this stage your airplane is empty. You have to define the top level requirements
and the general arrangement of the airplane as follow ::

    arrangement = Arrangement()  # default settings here
    requirement = Requirement()  # default settings again

You can also define your own settings. For example, the Top Level Aircraft Requirement can be specified as follow ::

    requirement =  Requirement(n_pax_ref = 150.,
                               design_range = unit.m_NM(3000.),  # convert Nautical Miles to meters
                               cruise_mach = 0.78,
                               cruise_altp = unit.m_ft(35000.))  # convert feets to meters


.. note:: MARILib works with the International System of units. But you can convert to many other units tanks to the module
   :mod:`marilib.utils.unit`. Add the following statement at the beginning of your script ::

        from marilib.utils import unit

To intialise your aircraft you need to run :func:`aircraft_root.Aircraft.factory`::

    myplane.factory(agmt, reqs)  # WARNING : arrangement must not be changed after this line

This will initialize all secondary requirements and build the airframe from a set of physical components.
You can still change the requirements, however **at this stage the arrangement is fixed** and can not be changed anymore.


Structure
=========

The airplane in MARILib is thought as an object that can flow through different design routines.

.. figure:: ../uml/aircraft.png
    :width: 100%
    :align: center
    :alt: UML aircraft package

    Simplified Class diagram of the aircraft package

Design drivers
--------------

The design is driven by user inputs :

*   The requirements defined by :class:`requirement.Requirement`.
*   The arrangement defined by :class:`aircraft_root.Arrangement`

Design parameters
-----------------

Most airplane design parameteers are stored in the airframe. The airframe is a list of physical components (wing, thrusters
fuselage ...) available in :mod:`marilib.aircraft.airframe.component` and in :mod:`marilib.aircraft.airframe.propulsion`.
Each component has:

* a positon and a geometry (fuselage lentgh, wing size, thruster diameter ...)
* aerodynamic length, surface and form factors if necessary.
* a mass and a center of gravity

.. todo:: In Marilib 2.0, the inertia tensor is implemented but not used yet.

Models
------

In order to run design or optimisation routines (MDA/MDO), the aircraft contains several models:

.. todo:: the documentation of this section is not finished.

Customizing your Aircraft
=========================

You can build new Aircraft, with new components that are not implemented yet.
To do so :

1.  add your new component in :mod:`marilib.aircraft.airframe.component`. Your component must inherit the abstract
    class :class:`marilib.aircraft.airframe.component.Component`.

2.  Implement the following pre-design functions for your component:

    * :func:`airframe.component.Component.eval_geometry`
    * :func:`airframe.component.Component.eval_mass`

3.  Change the factory of the aircraft (:func:`marilib.aircraft.aircraft_root.Aircraft.factory`) in order to add your
    component to the airframe.

4.  Whisper a prayer, and click the run button.

Submodules documentation
========================

"""

from marilib.aircraft.aircraft_root import Arrangement, Aircraft
from marilib.aircraft.requirement import Requirement
from marilib.aircraft.design.process import mda, mdf, draw_design_space,explore_design_space