#!/usr/bin/env python
"""
python_umat_bridge.py

This module defines a Python version of the UMAT routine that will be
invoked by the MPI process (e.g. on rank 1) to handle data exchanged
with Abaqus’s Fortran UMAT. The function python_umat() waits for incoming
messages containing the strain and material properties, computes the
elastic stress and tangent, and returns the result.
"""

from mpi4py import MPI
import numpy as np

def compute_elastic_response(strain, E, nu):
    """
    Compute the elastic response for a 3D isotropic material.

    Parameters:
        strain : (6,) ndarray
            The strain vector components [ε₁, ε₂, ε₃, γ₁₂, γ₂₃, γ₃₁].
        E : float
            Young's modulus.
        nu : float
            Poisson's ratio.

    Returns:
        stress : (6,) ndarray
            The computed stress vector.
        tangent : (6,6) ndarray
            The 3D elastic stiffness (tangent) matrix.
    """
    # Compute Lame parameters
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    mu  = E / (2.0 * (1.0 + nu))
    
    # Assemble the elastic stiffness matrix D.
    D = np.zeros((6, 6), dtype=np.float64)
    D[0, 0] = lam + 2.0 * mu;  D[0, 1] = lam;           D[0, 2] = lam
    D[1, 0] = lam;            D[1, 1] = lam + 2.0 * mu;   D[1, 2] = lam
    D[2, 0] = lam;            D[2, 1] = lam;            D[2, 2] = lam + 2.0 * mu
    D[3, 3] = mu
    D[4, 4] = mu
    D[5, 5] = mu

    # Compute the stress as: stress = D * strain.
    stress = np.dot(D, strain)
    return stress, D

def python_umat():
    """
    A Python version of the UMAT function.

    This routine waits in a loop for incoming data from Abaqus (sent by a Fortran UMAT
    running on MPI rank 0). The incoming message is expected to be a list of eight numbers:
      - The first six numbers are the strain components.
      - The next two numbers are the material properties (E and nu).
    
    The function computes the stress and the elastic stiffness matrix (tangent) using
    a 3D isotropic elastic model, packs these into a response array (6 stress values followed
    by 36 tangent values, flattened in row‐major order), and sends the reply back to Abaqus.

    To exit the loop, Abaqus (or another controller) may send a special message (here,
    we assume that sending `None` is the exit flag).
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    if rank != 1:
        raise RuntimeError("python_umat() should be executed on MPI rank 1.")

    print("Python UMAT process starting on rank 1. Waiting for UMAT calls from Abaqus ...")
    while True:
        # Wait for data from Abaqus (sent from rank 0 with tag 100)
        data = comm.recv(source=0, tag=100)
        # Use a special message (None) as an exit flag.
        if data is None:
            print("Received termination signal. Exiting python_umat loop.")
            break

        # The incoming data should be 8 numbers: 6 strains and 2 material props.
        data_arr = np.array(data, dtype=np.float64)
        if data_arr.size != 8:
            raise ValueError("Received data does not have 8 entries; got %d" % data_arr.size)
        strain = data_arr[0:6]
        E = data_arr[6]
        nu = data_arr[7]

        # Compute the elastic response.
        stress, tangent = compute_elastic_response(strain, E, nu)

        # Pack the reply:
        #   - First 6 numbers: stress components.
        #   - Next 36 numbers: 6x6 tangent matrix (flattened in row-major order).
        response = np.concatenate((stress, tangent.flatten(order='C')))
        # Send the response back to Abaqus (to rank 0, with tag 101)
        comm.send(response.tolist(), dest=0, tag=101)

if __name__ == '__main__':
    # If this module is executed as a script on rank 1, start the UMAT loop.
    python_umat()
