seed_value = 1

extra_variables_reco = [
    "gen_nu_p_1", "gen_nu_phi_1", "gen_nu_eta_1",  # leading neutrino, gen level
    "gen_nu_p_2", "gen_nu_phi_2", "gen_nu_eta_2"  # subleading neutrino, gen level
]
selectors_reco = [
    "wt_cp_sm", "wt_cp_ps", "wt_cp_mm", "rand",
    "mva_dm_1", "mva_dm_2",
    "tau_decay_mode_1", "tau_decay_mode_2",
]
selectors_gen = [
    "wt_cp_sm", "wt_cp_ps", "wt_cp_mm", "rand",
    "dm_1", "dm_2",
]
met_reco = [
    'metx', 'mety',
    'metcov00', 'metcov01', 'metcov10', 'metcov11'
]
met_gen = [
    'metx', 'mety'
]
ip = [
    'ip_x_1', 'ip_y_1', 'ip_z_1',
    'ip_x_2', 'ip_y_2', 'ip_z_2',
]
sv_1 = ['sv_x_1', 'sv_y_1', 'sv_z_1']
sv_2 = ['sv_x_2', 'sv_y_2', 'sv_z_2']
# variables are now fully loaded
particles_rho_rho = [
    "pi_E_1", "pi_px_1", "pi_py_1", "pi_pz_1",
    "pi0_E_1", "pi0_px_1", "pi0_py_1", "pi0_pz_1",
    "pi_E_2", "pi_px_2", "pi_py_2", "pi_pz_2",
    "pi0_E_2", "pi0_px_2", "pi0_py_2", "pi0_pz_2",
]
variables_rho_rho = [
    "aco_angle_1", "aco_angle_5", "aco_angle_6", "aco_angle_7",
    "y_1_1", "y_1_2",
    # "gen_nu_p_1", "gen_nu_phi_1", "gen_nu_eta_1",  # leading neutrino, gen level
    # "gen_nu_p_2", "gen_nu_phi_2", "gen_nu_eta_2"  # subleading neutrino, gen level
] + selectors_reco + met_reco + ip + particles_rho_rho + extra_variables_reco
particles_rho_a1 = [
    "pi_E_1", "pi_px_1", "pi_py_1", "pi_pz_1",
    "pi0_E_1", "pi0_px_1", "pi0_py_1", "pi0_pz_1",
    "pi_E_2", "pi_px_2", "pi_py_2", "pi_pz_2",
    "pi2_px_2", "pi2_py_2", "pi2_pz_2", "pi2_E_2",
    "pi3_px_2", "pi3_py_2", "pi3_pz_2", "pi3_E_2",
]
variables_rho_a1 = [
    "aco_angle_1", "aco_angle_2", "aco_angle_3", "aco_angle_4", 
    "y_1_1", "y_1_2", "y_2_2", "y_3_2", "y_4_2",
] + selectors_reco + met_reco + ip + sv_2 + particles_rho_a1
particles_a1_a1 = [
    "pi_E_1", "pi_px_1", "pi_py_1", "pi_pz_1",
    "pi_E_2", "pi_px_2", "pi_py_2", "pi_pz_2",
    "pi2_E_1", "pi2_px_1", "pi2_py_1", "pi2_pz_1",
    "pi3_E_1", "pi3_px_1", "pi3_py_1", "pi3_pz_1",
    "pi2_px_2", "pi2_py_2", "pi2_pz_2", "pi2_E_2",
    "pi3_px_2", "pi3_py_2", "pi3_pz_2", "pi3_E_2",
]
variables_a1_a1 = [
    "aco_angle_1", "aco_angle_2", "aco_angle_3", "aco_angle_4", 
    "y_1_1", "y_1_2", "y_2_2", "y_3_2", "y_4_2",
    'pv_angle',
] + selectors_reco + met_reco + ip + sv_1 + sv_2 + particles_a1_a1

variables_smearing = variables_rho_rho + [
    'reco_dm_1',
    'reco_pi_E_1', 'reco_pi_px_1', 'reco_pi_py_1', 'reco_pi_pz_1',
    'reco_pi0_E_1', 'reco_pi0_px_1', 'reco_pi0_py_1', 'reco_pi0_pz_1'
]

variables_gen_rho_rho = particles_rho_rho + selectors_gen + met_gen + sv_1 + sv_2

variables_gen_rho_a1 = particles_rho_a1 + selectors_gen + met_gen + sv_1 + sv_2

variables_gen_a1_a1 = particles_a1_a1 + selectors_gen + met_gen + sv_1 + sv_2
