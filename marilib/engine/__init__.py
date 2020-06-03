"""This subpackage is a wrapper for :mod:`marilib.engine.ExergeticEngine`, not developped by the MARILib Team.
It provides an exergetic approach to define the performances of various type of aircraft thrusters.
The parent class :class:`ExergeticEngine.ExergeticEngine` defines common features for the following thrusters :

* :class:`ExergeticEngine.Turbofan`
* :class:`ExergeticEngine.Turboprop`
* :class:`ExergeticEngine.ElectricFan`

The interface :mod:`marilib.engine.interface` provides tools to integrate this approach in MARILib.
"""