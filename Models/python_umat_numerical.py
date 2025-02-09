#!/usr/bin/env python
"""
python_umat_numerical.py

This module defines a Python version of the UMAT function that
communicates with Abaqus via MPI. It receives, on MPI rank 1,
an incoming message with 8 numbers:
    - 6 strain components (assumed ordered as: ε₁, ε₂, ε₃, γ₁₂, γ₂₃, γ₃₁)
    - 2 material properties: Young's modulus (E) and Poisson's ratio (ν)

It then computes:
    1. The elastic stress: stress = D * strain, where D is the elastic stiffness matrix.
    2. The tangent (Jacobian) d(stress)/d(strain) by numerical differentiation.
       For each strain component i the central difference approximation is used:
           d(stress)/d(strain_i) ≈ [stress(strain + δ_i) - stress(strain - δ_i)]/(2 δ)
       with δ set to a small number (here 1e–8).

Finally, the response (6 stress components followed by the flattened 6×6 tangent)
is sent back (via MPI tag 101) to the calling Fortran UMAT (running on rank 0).

Usage:
    Ensure mpi4py is installed and run this MPI job with at least 2 processes.
    This routine should be executed on MPI rank 1.
"""

from mpi4py import MPI
import numpy as np

def compute_stress(strain, E, nu):
    """
    Compute the stress vector for a 3D isotropic elastic material.
    
    Parameters:
        strain : (6,) ndarray
            The strain vector [ε₁, ε₂, ε₃, γ₁₂, γ₂₃, γ₃₁].
        E : float
            Young's modulus.
        nu : float
            Poisson's ratio.
    
    Returns:
        stress : (6,) ndarray
            The computed stress vector.
    """
    # Compute Lame parameters.
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    mu  = E / (2.0 * (1.0 + nu))
    
    # Build the elastic stiffness matrix D.
    D = np.zeros((6, 6), dtype=np.float64)
    D[0, 0] = lam + 2.0 * mu;  D[0, 1] = lam;           D[0, 2] = lam
    D[1, 0] = lam;            D[1, 1] = lam + 2.0 * mu;   D[1, 2] = lam
    D[2, 0] = lam;            D[2, 1] = lam;            D[2, 2] = lam + 2.0 * mu
    D[3, 3] = mu
    D[4, 4] = mu
    D[5, 5] = mu
    
    # Compute the stress vector.
    stress = np.dot(D, strain)
    return stress

def compute_jacobian(strain, E, nu, epsilon=1e-8):
    """
    Compute the Jacobian (tangent) d(stress)/d(strain) numerically by central differences.
    
    Parameters:
        strain : (6,) ndarray
            The base strain vector.
        E : float
            Young's modulus.
        nu : float
            Poisson's ratio.
        epsilon : float, optional
            The finite-difference perturbation size (default 1e-8).
    
    Returns:
        jacobian : (6,6) ndarray
            The approximated tangent stiffness matrix.
    """
    base_stress = compute_stress(strain, E, nu)
    jacobian = np.zeros((6, 6), dtype=np.float64)
    
    # Loop over each strain component.
    for i in range(6):
        perturb = np.zeros(6, dtype=np.float64)
        perturb[i] = epsilon
        
        stress_plus = compute_stress(strain + perturb, E, nu)
        stress_minus = compute_stress(strain - perturb, E, nu)
        
        # Central difference approximation.
        jacobian[:, i] = (stress_plus - stress_minus) / (2.0 * epsilon)
    
    return jacobian

def python_umat():
    """
    Wait for incoming UMAT calls from Abaqus (via MPI) and process them.
    
    Communication protocol:
      - Incoming messages (from MPI rank 0, tag 100) are expected to be a list of 8 numbers:
            [ε₁, ε₂, ε₃, γ₁₂, γ₂₃, γ₃₁, E, nu]
      - A special message of None signals termination.
      - The reply is a list of 42 numbers:
            [σ₁, σ₂, σ₃, σ₄, σ₅, σ₆, (6×6 tangent flattened in row-major order)]
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    if rank != 1:
        raise RuntimeError("python_umat() should only be executed on MPI rank 1.")
    
    print("Python UMAT with numerical Jacobian is running on MPI rank 1 and waiting for data.")
    
    while True:
        # Receive incoming data from Abaqus (sent from rank 0 with tag 100).
        data = comm.recv(source=0, tag=100)
        
        # Check for termination signal.
        if data is None:
            print("Received termination signal. Exiting python_umat loop.")
            break
        
        data_arr = np.array(data, dtype=np.float64)
        if data_arr.size != 8:
            raise ValueError("Expected 8 entries (6 strain components and 2 material properties); got %d." % data_arr.size)
        
        # Unpack the strain and material properties.
        strain = data_arr[0:6]
        E = data_arr[6]
        nu = data_arr[7]
        
        # Compute the stress.
        stress = compute_stress(strain, E, nu)
        # Compute the tangent (Jacobian) numerically.
        tangent = compute_jacobian(strain, E, nu, epsilon=1e-8)
        
        # Pack the reply:
        #   - First 6 numbers: the stress vector.
        #   - Next 36 numbers: the 6×6 tangent matrix, flattened in row-major order.
        response = np.concatenate((stress, tangent.flatten(order='C')))
        
        # Send the response back to Abaqus (to rank 0, tag 101).
        comm.send(response.tolist(), dest=0, tag=101)

if __name__ == '__main__':
    # If this module is executed as a script on rank 1, start the UMAT loop.
    python_umat()
