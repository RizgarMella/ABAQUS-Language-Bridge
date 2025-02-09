# ABAQUS Language Bridge

**Version:** Modernised in 2024

---

## Disclaimer

**WARNING:** This project is experimental and is provided “as-is” with no warranty. It was modernised in 2024 to explore the possibility of writing material models in Python—leveraging symbolic manipulation and automatic (or numerical) differentiation for Jacobians—to extend Abaqus. Use at your own risk. I can no longer continue the experiment because I no longer have access to Abaqus.

---

## Overview

This project demonstrates an experimental framework that allows material models in Abaqus to be written in Python. Traditionally, Abaqus uses UMATs written in Fortran to define material behaviour. In this experiment, a Fortran UMAT acts solely as a communication “bridge” that transfers data via MPI to a running Python process, where the material model is implemented. The framework supports:

- **Elastic Material Model:**
  - Computes the stress as *σ = D : ε* (with *D* as the elastic stiffness matrix).
  - The tangent can be computed either directly or by numerical differentiation.

- **Plastic Material Model with Isotropic Hardening:**
  - Implements a small-strain, J₂ (von Mises) plasticity model with isotropic hardening using a radial return mapping algorithm.
  - Updates the stress, internal state (plastic strain tensor and effective plastic strain), and computes the consistent tangent (Jacobian) by numerical differentiation.

The communication between the Fortran UMAT (running within Abaqus) and the Python process is handled using MPI (via [mpi4py](https://mpi4py.readthedocs.io/)).

---

## Project Structure

### 1. Fortran UMAT Bridge (`umat_bridge.f`)

- A Fortran subroutine that is compiled and linked with Abaqus.
- It packs strain and material data from Abaqus into an array, sends the data via MPI (tag 100) to the Python process (expected on MPI rank 1), and waits for a reply containing the computed stress, tangent, and optionally state data.

### 2. Python UMAT Modules

#### a. Python UMAT Server (Basic Elastic Response) – `python_umat_bridge.py`

- Defines a function `python_umat()` that listens (on MPI rank 1) for messages from Abaqus.
- Unpacks incoming data (6 strain components, *E*, and *ν*), computes the elastic stress response using a 3D isotropic elastic model, and sends back the stress and elastic stiffness matrix.

#### b. Python UMAT with Numerical Differentiation – `python_umat_numerical.py`

- Extends the basic elastic UMAT by computing the tangent (Jacobian) via numerical differentiation (using a central-difference scheme).

#### c. Python UMAT for Isotropically Hardening Plasticity – `python_umat_plastic.py`

- Implements a small-strain, von Mises plasticity model with isotropic hardening.
- Receives 17 numbers:
  - 6 strain components,
  - *E*,
  - *ν*,
  - yield stress (*σ<sub>y</sub>*),
  - hardening modulus (*H*),
  - previous plastic strain (6 numbers),
  - and previous effective plastic strain.
- Computes a trial elastic response, checks the yield function, and if yielding occurs, performs a plastic correction.
- Updates the plastic strain and effective plastic strain, and returns the corrected stress.
- Computes the consistent tangent via numerical differentiation.
- Returns a response containing the updated stress, tangent, and updated state variables.

### 3. Abaqus Input Deck (`single_element.inp`)

- A sample Abaqus input file defining a single 8-node brick element using a user-defined material.
- The UMAT call in Abaqus triggers the Fortran bridge, which in turn communicates with the Python UMAT.

---

## Detailed Documentation

### Fortran UMAT Bridge (`umat_bridge.f`)

This Fortran subroutine serves as the communication bridge inside Abaqus. It:
- Packs the strain (first 6 components) and material properties (*E* and *ν*) into an array.
- Sends the array via MPI (tag 100) to the Python process (assumed to run on MPI rank 1).
- Waits for a reply containing the computed results (stress, tangent, and possibly updated state) via MPI (tag 101).
- Unpacks the received data into Abaqus’ internal arrays.

### Python UMAT Modules

#### Python UMAT Server (Basic Elastic Response) – `python_umat_bridge.py`

- Waits on MPI rank 1 for incoming messages from Abaqus.
- Computes the elastic response (*stress = D × strain*) and returns both the stress and the elastic stiffness matrix.

#### Python UMAT with Numerical Differentiation – `python_umat_numerical.py`

- Computes the elastic stress response.
- Uses a central-difference scheme to numerically approximate the consistent tangent (Jacobian) matrix by perturbing each strain component.

#### Python UMAT for Isotropically Hardening Plasticity – `python_umat_plastic.py`

- Implements a J₂ plasticity model with isotropic hardening:
  - Computes a trial elastic stress based on the total strain minus the previous plastic strain.
  - Extracts the deviatoric stress and evaluates the yield function:
    \[
      f_{\text{trial}} = \| s_{\text{trial}} \| - \sqrt{\frac{2}{3}} \times \Bigl(\sigma_y + H \times (\text{previous effective plastic strain})\Bigr)
    \]
  - If \( f_{\text{trial}} \le 0 \), the step is elastic; otherwise, a plastic correction is performed:
    - Computes a plastic multiplier (Δγ).
    - Updates the plastic strain and effective plastic strain.
    - Returns the corrected stress.
- Computes the consistent tangent via numerical differentiation.
- Returns a response containing the updated stress, tangent, and updated state variables.

### Abaqus Input Deck (`single_element.inp`)

A sample input file that:
- Defines a single eight-node brick element.
- Uses a user material that calls the UMAT.
- Applies appropriate boundary conditions and loads.
- Demonstrates how the Fortran bridge and Python UMAT interact during the analysis.

---

## Usage Instructions

1. **Compile and Link the Fortran UMAT:**
   - Compile `umat_bridge.f` and link it with the Abaqus executable as required by your installation.

2. **Set Up the Python Environment:**
   - Install **mpi4py** and **NumPy**.
   - Run the desired Python UMAT module on MPI rank 1. For example:
     ```bash
     mpiexec -n 2 python python_umat_plastic.py
     ```
     In this example, MPI rank 0 is used by Abaqus (via the Fortran UMAT), and rank 1 runs the Python UMAT.

3. **Run Abaqus:**
   - Execute the Abaqus analysis using the input deck `single_element.inp` ensuring that Abaqus loads the user subroutine containing the Fortran bridge.

4. **Termination:**
   - To terminate the Python UMAT loop, a termination message (`None`) must be sent from Abaqus (or your test harness).

---

## Experimental Nature & Limitations

- **Experimental Status:**
  - This is an experimental project aimed at exploring the possibility of multilingual UMATs for Abaqus.
  
- **Automatic Differentiation:**
  - Although the current implementation uses numerical differentiation to compute the tangent, the framework is designed to be extended to leverage symbolic manipulation and automatic differentiation.
  
- **Risk Warning:**
  - This code is provided without warranty and is intended solely for experimental purposes. Use at your own risk.
  
- **Abaqus Access:**
  - I can no longer continue the experiment because I no longer have access to Abaqus. Future support and enhancements are not guaranteed.


