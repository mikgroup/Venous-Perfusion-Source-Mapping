# Venous Perfusion Source Mapping
Code for reproducing paper "MR Perfusion Source Mapping "In Reverse": Maps Venous Territories and Reveals Perfusion Modulation during Neural Activation"

## Description

The cerebral venous system is pivotal in various neurological and vascular conditions, as well as in regulating blood flow to support activated brain regions. Compared to the arterial system, the venous system hemodynamics is relatively unexplored due to its complexity and variability across individuals. To address this, we develop a venous perfusion source mapping method using Displacement Spectrum MRI, a non-contrast method that uses blood water as an endogenous contrast agent. Our technique encodes spatial information into the magnetization of blood water spins during tagging and remotely detects it once the tagged blood reaches the imaging region -- often near the brain's surface, where the signal-to-noise ratio is 3-4$\times$ higher. Through repeated spin-tagging and encoding, we can resolve the sources of blood water entering the imaging slice across short (10ms) to long (3s) evolution times, effectively capturing venous perfusion sources in reverse. Blood sources can be traced regardless of their path and velocity, enabling measurement of slow blood flow in smaller veins and potentially in capillary beds. In this work, we demonstrate perfusion source mapping in the superior cerebral veins, verify the sensitivity to global perfusion modulation induced by caffeine, and establish the specificity by showing consistent and repeatable local perfusion modulation due to neural activation. Remarkably, from all the blood present within veins in the imaging slice, our method can sense and localize the portion that originates from an activated region upstream.

## Instructions

The notebook <code>Reproduce Figures.ipynb</code> reproduces the figures and processing from the manuscript. Currently code to reproduce Figure 3 has been implemented, further code is coming up!
