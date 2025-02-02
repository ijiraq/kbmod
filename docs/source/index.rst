.. KBMoD documentation master file, created by
   sphinx-quickstart on Tue Nov 22 22:00:00 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. image:: _static/kbmod.svg
  :width: 400
  :alt: KBMoD logo

KBMOD (Kernel Based Moving Object Detection) is a GPU-accelerated framework for the detection of slowly moving asteroids within sequences of images. KBMOD enables users to detect moving objects that are too dim to be detected in a single image without requiring source detection in any individual image, nor at any fixed image cadence. KBMOD achieves this by “shift-and-stacking” images for a range of asteroid velocities and orbits without requiring the input images to be transformed to correct for the asteroid motion.

.. Important:: If you use KBMoD for work presented in a publication or talk please help the project via proper citation or acknowledgement **what is proper citation or acknowledgement**

	       
Getting Started
===============

.. toctree::
   :maxdepth: 1

   overview/overview
   overview/input_files
   overview/masking
   overview/search_params
   overview/output_files
   overview/results_filtering
   overview/testing

.. This then should be whatever else we want it to and does not need to be a dry list of all automodule commands

User Documentation
==================

.. toctree::
   :maxdepth: 1
	      
   run_search_referenceapi
   search_referenceapi
   analysis_utils
   evaluate
   image_info
   jointfit_functions
   kbmod_info
   kbmodpy
   known_objects


For Developers
==============



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
