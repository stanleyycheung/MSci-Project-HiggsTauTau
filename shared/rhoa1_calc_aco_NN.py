
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import uproot
from pylorentz import Momentum4
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score
import tensorflow as tf


def calc_aco_angles(pp1, pp2, pp3, pp4, yy1, yy2):
    angles = []
    for i in range(len(pp1)):
        p3 = pp3[i]
        p4 = pp4[i]
        p1 = pp1[i]
        p2 = pp2[i]
        y1 = yy1[i]
        y2 = yy2[i]

        def unit(vect):
            return vect / np.linalg.norm(vect)
        
        n1 = p1[1:] - np.dot(p1[1:], unit(p3[1:])) * unit(p3[1:])
        n2 = p2[1:] - np.dot(p2[1:], unit(p4[1:])) * unit(p4[1:])
        n1 = unit(n1)
        n2 = unit(n2)

        angle = np.arccos(np.dot(n1, n2))
        # print(p4.shape)
        # print(n1.shape)
        # print(n2.shape)
        sign = np.dot(unit(p4[1:]), np.cross(n1, n2))

        # shift 1
        if sign < 0:
            angle = 2 * np.pi - angle

        # shift 2
        if y1*y2 < 0:
            if angle < np.pi:
                angle += np.pi
            else:
                angle -= np.pi

        angles.append(angle)

        if i%100000==0:
            print('finished element', i)
            
    return angles


def calc_aco_angles_alie(pp1, pp2, pp3, pp4, yy1, yy2):
    pp1 = pp1.T
    pp2 = pp2.T
    pp3 = pp3.T
    pp4 = pp4.T
    
    #Some geometrical functions
    def cross_product(vector3_1,vector3_2):
        if len(vector3_1)!=3 or len(vector3_1)!=3:
            print('These are not 3D arrays !')
        x_perp_vector=vector3_1[1]*vector3_2[2]-vector3_1[2]*vector3_2[1]
        y_perp_vector=vector3_1[2]*vector3_2[0]-vector3_1[0]*vector3_2[2]
        z_perp_vector=vector3_1[0]*vector3_2[1]-vector3_1[1]*vector3_2[0]
        return np.array([x_perp_vector,y_perp_vector,z_perp_vector])
    
    def dot_product(vector1,vector2):
        if len(vector1)!=len(vector2):
            print('vector1 =', vector1)
            print('vector2 =', vector2)
            raise Exception('Arrays_of_different_size')
        prod=0
        for i in range(len(vector1)):
            prod=prod+vector1[i]*vector2[i]
        return prod

    def norm(vector):
        if len(vector)!=3:
            print('This is only for a 3d vector')
        return np.sqrt(vector[0]**2+vector[1]**2+vector[2]**2)
    
    #calculating the perpependicular component
    pi0_1_3Mom_star_perp=cross_product(pp1[1:], pp3[1:])
    pi0_2_3Mom_star_perp=cross_product(pp2[1:], pp4[1:])
    
    #Now normalise:
    pi0_1_3Mom_star_perp=pi0_1_3Mom_star_perp/norm(pi0_1_3Mom_star_perp)
    pi0_2_3Mom_star_perp=pi0_2_3Mom_star_perp/norm(pi0_2_3Mom_star_perp)
    
    #Calculating phi_star
    phi_CP=np.arccos(dot_product(pi0_1_3Mom_star_perp,pi0_2_3Mom_star_perp))
    
    
    #The energy ratios
    y_T = np.array(yy1 * yy2)
    
    #Up to here I agree with Kingsley
    print(phi_CP[:10],'\n')
    
    #The O variable
    cross=np.cross(pi0_1_3Mom_star_perp.transpose(),pi0_2_3Mom_star_perp.transpose()).transpose()
    bigO=dot_product(pp4[1:],cross)
    
    #perform the shift w.r.t. O* sign
    phi_CP=np.where(bigO>=0, 2*np.pi-phi_CP, phi_CP)#, phi_CP)
    
    #additionnal shift that needs to be done do see differences between odd and even scenarios, with y=Energy ratios
    #phi_CP=np.where(y_T<0, 2*np.pi-phi_CP, np.pi-phi_CP)
    phi_CP=np.where(y_T>=0, np.where(phi_CP<np.pi, phi_CP+np.pi, phi_CP-np.pi), phi_CP)

    return phi_CP

def calculate_aco_angles(pi_1, pi_2, pi0_1, pi2_2, pi3_2, y1, y2, which_aco_angle='rhoa1-5'):
    p3 = Momentum4(pi_1[:, 0], pi_1[:, 1], pi_1[:, 2], pi_1[:, 3]) # p3 = charged pion 1
    p4 = Momentum4(pi_2[:, 0], pi_2[:, 1], pi_2[:, 2], pi_2[:, 3]) # p4 = charged pion 2
    
    if which_aco_angle == 'rhoa1-5': # this is the same though process as rhoa1-4, but I think it's corrected, because particle 1 and particle 3 are the same composite particle
        pi0 = Momentum4(pi0_1[:, 0], pi0_1[:, 1], pi0_1[:, 2], pi0_1[:, 3]) # pi0 = neutral pion 1
        pi2 = Momentum4(pi2_2[:, 0], pi2_2[:, 1], pi2_2[:, 2], pi2_2[:, 3]) # pi2 = second charged pion 2
        pi3 = Momentum4(pi3_2[:, 0], pi3_2[:, 1], pi3_2[:, 2], pi3_2[:, 3]) # pi3 = third carged pion 3
        # p3 = pi_1
        # p4 = pi_2
        
        # # this gives: good distr for the p4+pi3 combination, but bad distr for the p4+pi2 neutral rho
        # p1 = p3
        # p2 = p4 + pi3
        # p3 = pi0
        # p4 = pi2
        
        p1 = pi0
        p2 = pi2
        p3 = p3
        p4 = p4 + pi3
        
        # # this is the other option:
        # p1 = p3
        # p2 = p4 + pi2
        # p3 = pi0
        # p4 = pi3
    
    return calc_aco_angles(p1[:].T, p2[:].T, p3[:].T, p4[:].T, y1, y2)


def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = np.asarray(axis)
    axis = axis / np.dot(axis, axis)**0.5
    a = math.cos(theta / 2.0)
    b, c, d = -axis * math.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])

def rotate(vect, axis, theta):
    return np.dot(rotation_matrix(axis, theta), vect)

def kristof_model(dimensions):
    # model by kristof
    model = tf.keras.models.Sequential()
    layers = 2
    model_str = "kristof_model"
    metrics = ['AUC', 'accuracy']
    model.add(tf.keras.layers.Dense(300, input_dim=dimensions, kernel_initializer='normal', activation='relu'))
    model.add(tf.keras.layers.Dense(300, kernel_initializer='normal', activation='relu'))
    model.add(tf.keras.layers.Dense(1, activation="sigmoid"))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=metrics)
    return model

def plotROCCurve(fpr, tpr, auc):
    #  define a function to plot the ROC curves - just makes the roc_curve look nicer than the default
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr)
    ax.set(xlabel='False Positive Rate', ylabel='True Positive Rate')
    ax.grid()
    ax.text(0.6, 0.3, 'Custom AUC Score: {:.3f}'.format(auc),
            bbox=dict(boxstyle='square,pad=0.3', fc='white', ec='k'))
    lims = [np.min([ax.get_xlim(), ax.get_ylim()]), np.max([ax.get_xlim(), ax.get_ylim()])]
    ax.plot(lims, lims, 'k--')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    # plt.savefig(f'{self.save_dir}/fig/ROC_curve_{self.config_str}.PNG')



# ============================== END OF FUNCTIONS ===============================

if __name__ == "__main__":
    tree_tt = uproot.open("C:\\Users\\krist\\Downloads\\MVAFILE_ALLHiggs_tt_new.root")["ntuple"]
    variables = [
            "wt_cp_sm", "wt_cp_ps", "wt_cp_mm", "rand",
            "aco_angle_1",
            "mva_dm_1", "mva_dm_2",
            "tau_decay_mode_1", "tau_decay_mode_2",
            "pi_E_1", "pi_px_1", "pi_py_1", "pi_pz_1",
            "pi_E_2", "pi_px_2", "pi_py_2", "pi_pz_2",
            "pi0_E_1", "pi0_px_1", "pi0_py_1", "pi0_pz_1",
            "pi2_px_2", "pi2_py_2", "pi2_pz_2", "pi2_E_2",
            "pi3_px_2", "pi3_py_2", "pi3_pz_2", "pi3_E_2",
            "ip_x_1", "ip_y_1", "ip_z_1",
            "sv_x_2", "sv_y_2", "sv_z_2",
            "y_1_1", "y_1_2",
        ]
    print('started reading in df')
    df = tree_tt.pandas.df(variables)
    print('finished reading in df')
    df_clean = df[(df['mva_dm_1']==1) & (df['mva_dm_2']==10) & (df["tau_decay_mode_1"] == 1)]
    df_ps = df_clean[(df_clean["rand"] < df_clean["wt_cp_ps"]/2)]
    df_sm = df_clean[(df_clean["rand"] < df_clean["wt_cp_sm"]/2)]
    len_df_ps = len(df_ps)
    
    y_sm = pd.DataFrame(np.ones(df_sm.shape[0]))
    y_ps = pd.DataFrame(np.zeros(df_ps.shape[0]))
    y = pd.concat([y_sm, y_ps]).to_numpy()
    df = pd.concat([df_sm, df_ps])
    
    pi_1 = Momentum4(df['pi_E_1'], df["pi_px_1"], df["pi_py_1"], df["pi_pz_1"])
    pi_2 = Momentum4(df['pi_E_2'], df["pi_px_2"], df["pi_py_2"], df["pi_pz_2"])
    pi0_1 = Momentum4(df['pi0_E_1'], df["pi0_px_1"], df["pi0_py_1"], df["pi0_pz_1"])
    pi2_2 = Momentum4(df['pi2_E_2'], df["pi2_px_2"], df["pi2_py_2"], df["pi2_pz_2"])
    pi3_2 = Momentum4(df['pi3_E_2'], df["pi3_px_2"], df["pi3_py_2"], df["pi3_pz_2"])
    rho_1 = pi_1 + pi0_1 # charged rho
    rho_2 = pi_2 + pi3_2 # neutral rho, a part of the charged a1 particle
    a1 = rho_2 + pi2_2
    # boost into rest frame of resonances
    rest_frame = pi_1 + pi_2 + pi0_1 + pi2_2 + pi3_2
    # rest_frame = pi0_1 + pi_1 + pi_2
    # rest_frame = pi_1 + pi_2
    boost = Momentum4(rest_frame[0], -rest_frame[1], -rest_frame[2], -rest_frame[3])
    # boost = - rest_frame
    pi_1_boosted = pi_1.boost_particle(boost)
    pi_2_boosted = pi_2.boost_particle(boost)
    pi0_1_boosted = pi0_1.boost_particle(boost)
    pi2_2_boosted = pi2_2.boost_particle(boost)
    pi3_2_boosted = pi3_2.boost_particle(boost)
    rho_1_boosted = pi_1_boosted + pi0_1_boosted
    rho_2_boosted = pi_2_boosted + pi3_2_boosted
    a1_boosted = rho_2_boosted + pi2_2_boosted
    rest_frame_boosted = pi_1_boosted + pi_2_boosted + pi0_1_boosted
    # rest_frame_boosted = rest_frame.boost_particle(boost)
    
    want_rotations = True # !!! Maybe this should be an input parameter
    
    # rotations
    if want_rotations:
        pi_1_boosted_rot, pi_2_boosted_rot = [], []
        pi0_1_boosted_rot, pi2_2_boosted_rot, pi3_2_boosted_rot = [], [], []
        rho_1_boosted_rot, rho_2_boosted_rot, a1_boosted_rot = [], [], []
        
        # MY ROTATIONS:
        # unit vectors along the momenta of the primary resonances
        unit1 = (rho_1_boosted[1:, :] / np.linalg.norm(rho_1_boosted[1:, :], axis=0)).transpose()
        unit2 = (pi_2_boosted[1:, :] / np.linalg.norm(pi_2_boosted[1:, :], axis=0)).transpose()
        # probably there's a faster way of doing this
        zaxis = np.array([np.array([0., 0., 1.]) for _ in range(len(unit1))])
        axes1 = np.cross(unit1, zaxis)
        axes2 = np.cross(unit2, zaxis)
        dotproduct1 = (unit1*zaxis).sum(1)
        angles1 = np.arccos(dotproduct1)
        dotproduct2 = (unit2*zaxis).sum(1)
        angles2 = np.arccos(dotproduct2)
        
        for i in range(pi_1_boosted[:].shape[1]):
# =============================================================================
#                 # STANLEY'S ROTATIONS:
#                 # rot_mat = self.rotation_matrix_from_vectors(rho_1_boosted[1:, i], [0, 0, 1])
#                 # rot_mat = self.rotation_matrix_from_vectors(pi_1_boosted[1:, i]+pi_2_boosted[1:, i], [0, 0, 1])
#                 # rot_mat = self.rotation_matrix_from_vectors(a1_boosted[1:, i], [0, 0, 1])
#                 rot_mat = self.rotation_matrix_from_vectors(rest_frame_boosted[1:, i], [0, 0, 1])
#                 pi_1_boosted_rot.append(rot_mat.dot(pi_1_boosted[1:, i]))
#                 pi0_1_boosted_rot.append(rot_mat.dot(pi0_1_boosted[1:, i]))
#                 pi_2_boosted_rot.append(rot_mat.dot(pi_2_boosted[1:, i]))
#                 pi2_2_boosted_rot.append(rot_mat.dot(pi2_2_boosted[1:, i]))
#                 pi3_2_boosted_rot.append(rot_mat.dot(pi3_2_boosted[1:, i]))
#                 rho_1_boosted_rot.append(rot_mat.dot(rho_1_boosted[1:, i]))
#                 rho_2_boosted_rot.append(rot_mat.dot(rho_2_boosted[1:, i]))
#                 a1_boosted_rot.append(rot_mat.dot(a1_boosted[1:, i]))
# =============================================================================
            
            # MY ROTATIONS:
            pi_1_boosted_rot.append(rotate(pi_1_boosted[1:, i], axes1[i], angles1[i]))
            pi0_1_boosted_rot.append(rotate(pi0_1_boosted[1:, i], axes1[i], angles1[i]))
            pi_2_boosted_rot.append(rotate(pi_2_boosted[1:, i], axes1[i], angles1[i]))
            pi2_2_boosted_rot.append(rotate(pi2_2_boosted[1:, i], axes1[i], angles1[i]))
            pi3_2_boosted_rot.append(rotate(pi3_2_boosted[1:, i], axes1[i], angles1[i]))
            rho_1_boosted_rot.append(rotate(rho_1_boosted[1:, i], axes1[i], angles1[i]))
            rho_2_boosted_rot.append(rotate(rho_2_boosted[1:, i], axes1[i], angles1[i]))
            a1_boosted_rot.append(rotate(a1_boosted[1:, i], axes1[i], angles1[i]))
            
            if i % 100000 == 0:
                print('finished getting rotated 4-vector', i)
        pi_1_boosted_rot = np.array(pi_1_boosted_rot)
        pi_2_boosted_rot = np.array(pi_2_boosted_rot)
        pi0_1_boosted_rot = np.array(pi0_1_boosted_rot)
        pi2_2_boosted_rot = np.array(pi2_2_boosted_rot)
        pi3_2_boosted_rot = np.array(pi3_2_boosted_rot)
        rho_1_boosted_rot = np.array(rho_1_boosted_rot)
        rho_2_boosted_rot = np.array(rho_2_boosted_rot)
        a1_boosted_rot = np.array(a1_boosted_rot)
        
        # # write out some rotated 4-vectors to a file, to compare with shared code
        # print('started writing out 4-vectors')
        # with open('4vectors/rotated_4vectors.txt', 'w') as f:
        #     for i in range(10):
        #         p1str = ' '.join([str(x) for x in pi0_1_boosted_rot[i]])
        #         p3str = ' '.join([str(x) for x in pi_1_boosted_rot[i]])
        #         p4str = ' '.join([str(x) for x in pi_2_boosted_rot[i]])
        #         f.write(p1str+'\t\t'+p3str+'\t\t'+p4str+'\n')
        # print('finished writing out 4-vectors')
        
        def padded(vect3):
            zeros = np.reshape(np.zeros(len(vect3)), (-1, 1))
            return np.concatenate([zeros, vect3], axis=1)
        # zeros_1 = np.reshape(np.zeros(len(pi_1_boosted_rot)), (-1, 1))
        # zeros_2 = np.reshape(np.zeros(len(pi_2_boosted_rot)), (-1, 1))
        # zeros_3 = np.reshape(np.zeros(len(pi0_1_boosted_rot)), (-1, 1))
        # zeros_4 = np.reshape(np.zeros(len(pi2_2_boosted_rot)), (-1, 1))
        # zeros_5 = np.reshape(np.zeros(len(pi3_2_boosted_rot)), (-1, 1))
        # aco_angle_2 = calculate_aco_angles(np.concatenate([zeros_1, pi_1_boosted_rot], axis=1), np.concatenate([zeros_1, pi_2_boosted_rot], axis=1), np.concatenate([zeros_1, pi0_1_boosted_rot], axis=1), np.concatenate([zeros_1, pi2_2_boosted_rot], axis=1), np.concatenate([zeros_1, pi3_2_boosted_rot], axis=1), df['y_1_1'].to_numpy(), df['y_1_2'].to_numpy())
        
        # aco_angle_2 = calculate_aco_angles(padded(pi_1_boosted_rot), padded(pi_2_boosted_rot), padded(pi0_1_boosted_rot), padded(pi2_2_boosted_rot), padded(pi3_2_boosted_rot), df['y_1_1'].to_numpy(), df['y_1_2'].to_numpy())
        # aco_angle_danny = calc_aco_angles(padded(pi0_1_boosted_rot[:]), padded(pi_2_boosted_rot[:]), padded(pi_1_boosted_rot[:]), padded(pi2_2_boosted_rot[:]), df['y_1_1'].to_numpy(), df['y_1_2'].to_numpy())
        print('shape =', pi0_1_boosted_rot[:].shape)
        aco_angle_danny = calc_aco_angles_alie(padded(pi0_1_boosted_rot[:]), padded(pi_2_boosted_rot[:]), padded(pi_1_boosted_rot[:]), padded(pi2_2_boosted_rot[:]), df['y_1_1'].to_numpy(), df['y_1_2'].to_numpy())
        aco_angle_2 = aco_angle_danny
        
        with open('rhoa1_aco_angle_calc.txt', 'w') as f:
            f.write('\n'.join([str(x) for x in aco_angle_2[-20:]]))
        with open('rhoa1_aco_angle_given.txt', 'w') as f:
            f.write('\n'.join([str(x) for x in df['aco_angle_1'].to_numpy()[-20:]]))
        print(np.sum(np.abs(aco_angle_2 - df['aco_angle_1']) < 0.01))
        print(np.sum(np.array(aco_angle_2) < np.inf))
        print(np.sum(np.isnan(aco_angle_2)))
        print((np.array(aco_angle_2) == np.inf).any())
        nanmask = np.isnan(aco_angle_2)
        aco_angle_2 = np.array(aco_angle_2)
        aco_angle_2[nanmask] = np.pi
        
        plt.figure(12)
        aco_angle_2_ps = aco_angle_2[:len_df_ps]
        aco_angle_2_sm = aco_angle_2[len_df_ps:]
        plt.hist(aco_angle_2_ps, bins=50, alpha=0.5)
        plt.hist(aco_angle_2_sm, bins=50, alpha=0.5)
        
    else: # if don't want rotations:
        pi_1_boosted_rot = np.array(pi_1_boosted).T
        pi_2_boosted_rot = np.array(pi_2_boosted).T
        pi0_1_boosted_rot = np.array(pi0_1_boosted).T
        pi2_2_boosted_rot = np.array(pi2_2_boosted).T
        pi3_2_boosted_rot = np.array(pi3_2_boosted).T
        rho_1_boosted_rot = np.array(rho_1_boosted).T
        rho_2_boosted_rot = np.array(rho_2_boosted).T
        a1_boosted_rot = np.array(a1_boosted).T
        
    df_inputs_data = {
        'pi_E_1_br': pi_1_boosted[0],
        'pi_px_1_br': pi_1_boosted_rot[:, 0],
        'pi_py_1_br': pi_1_boosted_rot[:, 1],
        'pi_pz_1_br': pi_1_boosted_rot[:, 2],
        'pi_E_2_br': pi_2_boosted[0],
        'pi_px_2_br': pi_2_boosted_rot[:, 0],
        'pi_py_2_br': pi_2_boosted_rot[:, 1],
        'pi_pz_2_br': pi_2_boosted_rot[:, 2],
        'pi0_E_1_br': pi0_1_boosted[0],
        'pi0_px_1_br': pi0_1_boosted_rot[:, 0],
        'pi0_py_1_br': pi0_1_boosted_rot[:, 1],
        'pi0_pz_1_br': pi0_1_boosted_rot[:, 2],
        'pi2_E_2_br': pi2_2_boosted[0],
        'pi2_px_2_br': pi2_2_boosted_rot[:, 0],
        'pi2_py_2_br': pi2_2_boosted_rot[:, 1],
        'pi2_pz_2_br': pi2_2_boosted_rot[:, 2],
        'pi3_E_2_br': pi3_2_boosted[0],
        'pi3_px_2_br': pi3_2_boosted_rot[:, 0],
        'pi3_py_2_br': pi3_2_boosted_rot[:, 1],
        'pi3_pz_2_br': pi3_2_boosted_rot[:, 2],
        'rho_E_1_br': rho_1_boosted[0],
        'rho_px_1_br': rho_1_boosted_rot[:, 0],
        'rho_py_1_br': rho_1_boosted_rot[:, 1],
        'rho_pz_1_br': rho_1_boosted_rot[:, 2],
        'rho_E_2_br': rho_2_boosted[0],
        'rho_px_2_br': rho_2_boosted_rot[:, 0],
        'rho_py_2_br': rho_2_boosted_rot[:, 1],
        'rho_pz_2_br': rho_2_boosted_rot[:, 2],
        'a1_E_br': a1_boosted[0],
        'a1_px_br': a1_boosted_rot[:, 0],
        'a1_py_br': a1_boosted_rot[:, 1],
        'a1_pz_br': a1_boosted_rot[:, 2],
        # 'aco_angle_1': df['aco_angle_1'],
        'aco_angle_1': aco_angle_2,
        'y_1_1': df['y_1_1'],
        'y_1_2': df['y_1_2'],
        'w_a': df.wt_cp_sm,
        'w_b': df.wt_cp_ps,
        'm_1': rho_1.m,
        #'m_2': rho_2.m,
        'm_2': a1.m,
    }
    
    df = pd.DataFrame(df_inputs_data)
    df['y'] = y
    
    # X_train, X_test, y_train, y_test = self.configure(df, config_num)
    pi_1_transformed = np.c_[df.pi_E_1_br, df.pi_px_1_br, df.pi_py_1_br, df.pi_pz_1_br, ]
    pi_2_transformed = np.c_[df.pi_E_2_br, df.pi_px_2_br, df.pi_py_2_br, df.pi_pz_2_br, ]
    pi0_1_transformed = np.c_[df.pi0_E_1_br, df.pi0_px_1_br, df.pi0_py_1_br, df.pi0_pz_1_br, ]
    pi2_2_transformed = np.c_[df.pi2_E_2_br, df.pi2_px_2_br, df.pi2_py_2_br, df.pi2_pz_2_br, ]
    pi3_2_transformed = np.c_[df.pi3_E_2_br, df.pi3_px_2_br, df.pi3_py_2_br, df.pi3_pz_2_br, ]
    
    config = 1 # !!! input parameter
    
    if config == 1:
        X = np.c_[df.aco_angle_1]
        # X = np.c_[df.aco_angle_1, df.aco_angle_2]
    if config == 2:
        X = np.c_[df.aco_angle_1, df.y_1_1, df.y_1_2]
    if config == 3:
        X = np.c_[pi0_1_transformed, pi2_2_transformed, pi3_2_transformed, pi_1_transformed, pi_2_transformed]
    if config == 4:
        # X = np.c_[pi_1_transformed, pi_2_transformed, pi0_1_transformed, pi2_2_transformed, pi3_2_transformed, self.df.aco_angle_1],
        X = np.c_[df.aco_angle_1, pi_1_transformed, pi_2_transformed, pi0_1_transformed] # reduced config 4 to exclude all variables not existing in rho-rho
    if config == 5:
        X = np.c_[df.aco_angle_1, df.y_1_1, df.y_1_2, df.m_1**2, df.m_2**2]
    if config == 6:
        X = np.c_[pi_1_transformed, pi_2_transformed, pi0_1_transformed, pi2_2_transformed, pi3_2_transformed, df.aco_angle_1, df.y_1_1, df.y_1_2, df.m_1**2, df.m_2**2]
    
    y = df['y']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123456, stratify=y)
    
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Training config {}~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'.format(config))
    model = kristof_model(X_train.shape[1])
    
    # TRAINING
    patience = 10
    batch_size = 10000
    epochs = 10 # !!! input parameter
    history = tf.keras.callbacks.History()
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience)
    # log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    # tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    model.fit(X_train, y_train,
                   batch_size=batch_size,
                   epochs=epochs,
                   callbacks=[history, early_stop],
                   validation_data=(X_test, y_test),
                   verbose=1)
    
    # EVALUATION
    y_proba = model.predict(X_test)  # outputs two probabilties
    # print(y_proba)
    auc = roc_auc_score(y_test, y_proba)
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plotROCCurve(fpr, tpr, auc)
    
    