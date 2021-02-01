seed_value = 1

# variables are bare bones only for rho_rho
variables_rho_rho = [
    "wt_cp_sm", "wt_cp_ps", "wt_cp_mm", "rand",
    "aco_angle_1",
    "mva_dm_1", "mva_dm_2",
    "tau_decay_mode_1", "tau_decay_mode_2",
    "pi_E_1", "pi_px_1", "pi_py_1", "pi_pz_1",
    "pi_E_2", "pi_px_2", "pi_py_2", "pi_pz_2",
    "pi0_E_1", "pi0_px_1", "pi0_py_1", "pi0_pz_1",
    "pi0_E_2", "pi0_px_2", "pi0_py_2", "pi0_pz_2",
    "y_1_1", "y_1_2",
    'met', 'metx', 'mety',
    'metcov00', 'metcov01', 'metcov10', 'metcov11',
    "gen_nu_p_1", "gen_nu_phi_1", "gen_nu_eta_1",  # leading neutrino, gen level
    "gen_nu_p_2", "gen_nu_phi_2", "gen_nu_eta_2"  # subleading neutrino, gen level
]
variables_rho_a1 = [
    "wt_cp_sm", "wt_cp_ps", "wt_cp_mm", "rand",
    "aco_angle_1",
    "mva_dm_1", "mva_dm_2",
    "tau_decay_mode_1", "tau_decay_mode_2",
    "pi_E_1", "pi_px_1", "pi_py_1", "pi_pz_1",
    "pi0_E_1", "pi0_px_1", "pi0_py_1", "pi0_pz_1",
    "pi_E_2", "pi_px_2", "pi_py_2", "pi_pz_2",
    "pi2_px_2", "pi2_py_2", "pi2_pz_2", "pi2_E_2",
    "pi3_px_2", "pi3_py_2", "pi3_pz_2", "pi3_E_2",
    "ip_x_1", "ip_y_1", "ip_z_1",
    "sv_x_2", "sv_y_2", "sv_z_2",
    "y_1_1", "y_1_2",
]
variables_a1_a1 = [
    "wt_cp_sm", "wt_cp_ps", "wt_cp_mm", "rand",
    "aco_angle_1",
    "mva_dm_1", "mva_dm_2",
    "tau_decay_mode_1", "tau_decay_mode_2",
    "pi_E_1", "pi_px_1", "pi_py_1", "pi_pz_1",
    "pi_E_2", "pi_px_2", "pi_py_2", "pi_pz_2",
    "pi2_E_1", "pi2_px_1", "pi2_py_1", "pi2_pz_1",
    "pi3_E_1", "pi3_px_1", "pi3_py_1", "pi3_pz_1",
    "pi2_px_2", "pi2_py_2", "pi2_pz_2", "pi2_E_2",
    "pi3_px_2", "pi3_py_2", "pi3_pz_2", "pi3_E_2",
    "ip_x_1", "ip_y_1", "ip_z_1",
    "sv_x_2", "sv_y_2", "sv_z_2",
    "y_1_1", "y_1_2",
]


# not checked yet
variables_gen_rho_rho = [
    "wt_cp_sm", "wt_cp_ps", "wt_cp_mm", "rand",
    "dm_1", "dm_2",
    "pi_E_1", "pi_px_1", "pi_py_1", "pi_pz_1",  # charged pion 1
    "pi_E_2", "pi_px_2", "pi_py_2", "pi_pz_2",  # charged pion 2
    "pi0_E_1", "pi0_px_1", "pi0_py_1", "pi0_pz_1",  # neutral pion 1
    "pi0_E_2", "pi0_px_2", "pi0_py_2", "pi0_pz_2",  # neutral pion 2,
    'metx', 'mety',
    'sv_x_1', 'sv_y_1', 'sv_z_1', 'sv_x_2', 'sv_y_2', 'sv_z_2', 
    ]

variables_gen_rho_rho = [
    "wt_cp_sm", "wt_cp_ps", "wt_cp_mm", "rand",
    "dm_1", "dm_2",
    "pi_E_1", "pi_px_1", "pi_py_1", "pi_pz_1",
    "pi0_E_1", "pi0_px_1", "pi0_py_1", "pi0_pz_1",
    "pi_E_2", "pi_px_2", "pi_py_2", "pi_pz_2",
    "pi2_px_2", "pi2_py_2", "pi2_pz_2", "pi2_E_2",
    "pi3_px_2", "pi3_py_2", "pi3_pz_2", "pi3_E_2",
    "sv_x_2", "sv_y_2", "sv_z_2",
]

variables_gen_a1_a1 = [
    "wt_cp_sm", "wt_cp_ps", "wt_cp_mm", "rand",
    "dm_1", "dm_2",
    "pi_E_1", "pi_px_1", "pi_py_1", "pi_pz_1",
    "pi_E_2", "pi_px_2", "pi_py_2", "pi_pz_2",
    "pi2_E_1", "pi2_px_1", "pi2_py_1", "pi2_pz_1",
    "pi3_E_1", "pi3_px_1", "pi3_py_1", "pi3_pz_1",
    "pi2_px_2", "pi2_py_2", "pi2_pz_2", "pi2_E_2",
    "pi3_px_2", "pi3_py_2", "pi3_pz_2", "pi3_E_2",
    "sv_x_1", "sv_y_1", "sv_z_1",
    "sv_x_2", "sv_y_2", "sv_z_2",
    "nu_E_1", "nu_px_1", "nu_py_1", "nu_pz_1", 
    "nu_E_2", "nu_px_2", "nu_py_2", "nu_pz_2",
    ]