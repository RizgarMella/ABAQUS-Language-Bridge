#!/usr/bin/env python
"""
python_umat_plastic.py

A Python version of a UMAT for small‐strain, J2 plasticity with isotropic hardening.
This routine uses a radial return mapping algorithm to update the stress and internal
state (plastic strain tensor and effective plastic strain). The consistent tangent is
computed by numerical differentiation.

Communication protocol (via MPI):
  Incoming message (from Abaqus’s Fortran UMAT on rank 0, tag 100) is expected to be
  an array of 17 numbers:
      [ε₁, ε₂, ε₃, γ₁₂, γ₂₃, γ₃₁, E, ν, σ_y, H, p_old₁, ..., p_old₆, ep_old]
  where:
      - ε_i are the strain components in Voigt notation,
      - E is Young’s modulus,
      - ν is Poisson’s ratio,
      - σ_y is the yield stress,
      - H is the isotropic hardening modulus,
      - p_old (6 numbers) is the previous plastic strain tensor,
      - ep_old is the previous effective (accumulated) plastic strain.
      
  The reply (sent to rank 0, tag 101) is an array of 49 numbers:
      [σ₁, ..., σ₆, (6×6 tangent flattened in row‐major order), (p_new (6 numbers), ep_new)]
      
Run this module on MPI rank 1.
"""

from mpi4py import MPI
import numpy as np

def elastic_stiffness(E, nu):
    """
    Assemble and return the 6x6 elastic stiffness matrix (in Voigt notation)
    for an isotropic material.
    """
    mu = E / (2.0*(1.0+nu))
    lam = E * nu / ((1.0+nu)*(1.0-2.0*nu))
    D = np.zeros((6,6), dtype=np.float64)
    D[0,0] = lam + 2.0*mu; D[0,1] = lam;         D[0,2] = lam
    D[1,0] = lam;         D[1,1] = lam + 2.0*mu; D[1,2] = lam
    D[2,0] = lam;         D[2,1] = lam;         D[2,2] = lam + 2.0*mu
    D[3,3] = mu
    D[4,4] = mu
    D[5,5] = mu
    return D, mu

def deviator(sigma):
    """
    Compute the deviatoric part of a stress vector in Voigt notation.
    Assumes that the first three components are normal stresses.
    """
    trace = sigma[0] + sigma[1] + sigma[2]
    hydro = trace / 3.0
    s = np.copy(sigma)
    s[0] -= hydro
    s[1] -= hydro
    s[2] -= hydro
    return s

def norm_deviator(s):
    """
    Compute the norm of a deviatoric stress in Voigt notation.
    Note that the shear components contribute with a factor of 2.
    """
    return np.sqrt(s[0]**2 + s[1]**2 + s[2]**2 + 2.0*(s[3]**2 + s[4]**2 + s[5]**2))

def compute_stress_plastic(strain, E, nu, sigy, H, statev_old):
    """
    Compute the updated stress and internal state for J2 plasticity with isotropic hardening.

    Parameters:
      strain      : (6,) ndarray, the total strain (Voigt notation)
      E, nu       : Elastic constants.
      sigy, H     : Yield stress and hardening modulus.
      statev_old  : (7,) ndarray, with the previous plastic strain tensor (first 6 numbers)
                    and the previous effective plastic strain (7th number).

    Returns:
      sigma     : (6,) ndarray, the updated stress.
      statev_new: (7,) ndarray, updated state variables (plastic strain tensor and effective plastic strain).
    """
    D, mu = elastic_stiffness(E, nu)
    # Extract previous plastic strain (p_old) and effective plastic strain (ep_old)
    p_old = statev_old[0:6]
    ep_old = statev_old[6]

    # Compute trial elastic strain and stress: ε_e_trial = ε_total - p_old, σ_trial = D : ε_e_trial
    eps_e_trial = strain - p_old
    sigma_trial = np.dot(D, eps_e_trial)
    
    # Compute deviatoric part and its norm
    s_trial = deviator(sigma_trial)
    norm_s_trial = norm_deviator(s_trial)
    
    # Evaluate yield function:
    # f_trial = ||s_trial|| - sqrt(2/3) * (sigy + H * ep_old)
    f_trial = norm_s_trial - np.sqrt(2.0/3.0)*(sigy + H*ep_old)
    
    if f_trial <= 0:
        # Elastic step: no update of plastic strain
        sigma = sigma_trial
        p_new = p_old
        ep_new = ep_old
    else:
        # Plastic step: compute plastic multiplier using radial return
        delta_gamma = f_trial / (2.0*mu + H*np.sqrt(2.0/3.0))
        # Unit normal vector in stress space (avoid division by zero)
        if norm_s_trial == 0:
            n = np.zeros(6)
        else:
            n = s_trial / norm_s_trial
        # Update plastic strain: p_new = p_old + Δγ * n
        p_new = p_old + delta_gamma * n
        # Update effective plastic strain: ep_new = ep_old + √(2/3)*Δγ
        ep_new = ep_old + np.sqrt(2.0/3.0)*delta_gamma
        # Return-mapped stress:
        sigma = sigma_trial - 2.0*mu*delta_gamma*n

    statev_new = np.concatenate((p_new, [ep_new]))
    return sigma, statev_new

def compute_jacobian_plastic(strain, E, nu, sigy, H, statev_old, epsilon=1e-8):
    """
    Compute the 6x6 tangent matrix (dσ/dε) numerically by central differences.

    The state update is part of the stress update so that the computed tangent is
    the consistent tangent.
    """
    base_sigma, _ = compute_stress_plastic(strain, E, nu, sigy, H, statev_old)
    jacobian = np.zeros((6,6), dtype=np.float64)
    for i in range(6):
        perturb = np.zeros(6, dtype=np.float64)
        perturb[i] = epsilon
        sigma_plus, _ = compute_stress_plastic(strain + perturb, E, nu, sigy, H, statev_old)
        sigma_minus, _ = compute_stress_plastic(strain - perturb, E, nu, sigy, H, statev_old)
        jacobian[:, i] = (sigma_plus - sigma_minus) / (2.0*epsilon)
    return jacobian

def python_umat():
    """
    Wait in a loop for incoming UMAT calls from Abaqus (via MPI) and process them.

    Communication protocol:
      - Incoming messages (from MPI rank 0, tag 100) are expected to be a list of 17 numbers:
            [ε₁,...,ε₆, E, ν, σ_y, H, p_old₁,...,p_old₆, ep_old]
      - A message of None signals termination.
      - The reply is a list of 49 numbers:
            [σ₁,...,σ₆, (6×6 tangent flattened row-major), (p_new₁,...,p_new₆, ep_new)]
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    if rank != 1:
        raise RuntimeError("python_umat() should be executed on MPI rank 1.")
    
    print("Python plastic UMAT is running on MPI rank 1. Waiting for UMAT calls ...")
    while True:
        data = comm.recv(source=0, tag=100)
        if data is None:
            print("Received termination signal. Exiting python_umat loop.")
            break
        
        data_arr = np.array(data, dtype=np.float64)
        if data_arr.size != 17:
            raise ValueError("Expected 17 entries; got %d." % data_arr.size)
        
        # Unpack the incoming data:
        strain    = data_arr[0:6]
        E         = data_arr[6]
        nu        = data_arr[7]
        sigy      = data_arr[8]
        H         = data_arr[9]
        statev_old = data_arr[10:17]  # first 6: plastic strain, 7th: effective plastic strain
        
        # Compute the stress update and state update.
        sigma, statev_new = compute_stress_plastic(strain, E, nu, sigy, H, statev_old)
        # Compute the consistent tangent by numerical differentiation.
        tangent = compute_jacobian_plastic(strain, E, nu, sigy, H, statev_old, epsilon=1e-8)
        
        # Pack the reply:
        #   - First 6 entries: updated stress.
        #   - Next 36 entries: tangent matrix (flattened in row-major order).
        #   - Final 7 entries: updated state variables.
        response = np.concatenate((sigma, tangent.flatten(order='C'), statev_new))
        comm.send(response.tolist(), dest=0, tag=101)

if __name__ == '__main__':
    python_umat()
