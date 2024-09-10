import numpy as np
from numba import njit
import os
import glob

from params_io import ParamData
from dataclasses import dataclass, field, fields

# start definition of dataclasses and default values of attributes ...

@dataclass
class MD_Params(ParamData):
    n_relax: int = 5_000
    n_observe: int = 25_000
    d_time: float = 0.1
    temperature: float = 0
    random_seed: int = 47_114_711

@dataclass
class SQE_Params(ParamData):
    r_cutoff: float = 1.5
    kappa_0: float = 2.4
    resist: float = 0.43 / 3.455
    pseudo_l: float = 1

    def d_time(self) -> float:
        d_time = 2 * np.pi * (self.pseudo_l/self.kappa_0)**0.5 / 40
        return min(d_time, self.pseudo_l / (20*self.resist + 1e-10) )

@dataclass
class RC_Params(ParamData):
    bond_length: float = 1.0
    capa_radius: float = 15.0   # approx. radius of capacitor plates (15)
    capa_dist:   float = 3.0    # distance between capacitor plates  (3)
    wire_length: float = 5.0    # length of wire normal to plates
    wire_height: float = 2*capa_radius  # origin-battery distance

@dataclass
class BattParams(ParamData):
    voltage: float = 1.0
    switch_open: bool = False

@dataclass
class IonInitParams(ParamData):
    r_y_0: float = None
    v_y_0: float = None

# ... end definition of dataclasses and their default values 

# start (re) definition of values contained in dataclasses ...

def charging():
    p_md = MD_Params(n_relax = 2_000, n_observe = 11_250) # Fig. 2
    p_sqe = SQE_Params()
    # to change from D-SQE to SQE-omega comment in next line
    # p_sqe = SQE_Params(resist = 0)
    p_batt = BattParams()
    p_rce = RC_Params()
    return p_md, p_sqe, p_batt, p_rce

def nyquist():
    p_md = MD_Params(temperature = 300./50_000, n_relax = 1_000, n_observe = 2_000)
    p_sqe = SQE_Params()
    p_batt = BattParams(voltage = 0)
    p_rce = RC_Params()
    return p_md, p_sqe, p_batt, p_rce

def friction():

    # it's best to define n_observe so that initial position defined in ion(...)
    # is the mirror image of the final position, i.e., satisfy
    # n_observe =  2 * r_y_0 / (d_time * v_y_0)
    p_md = MD_Params(n_relax = 5_000, n_observe = 12_000)
    p_sqe = SQE_Params()
    p_batt = BattParams(voltage = 0)
    p_rce = RC_Params()
    return p_md, p_sqe, p_batt, p_rce

def ion(p_rce):
    return IonInitParams(r_y_0 = -2*p_rce.capa_radius, v_y_0 = 0.05)

# ... end redefinition of values contained in dataclasses

def params_init(simul_mode):

    with open('params.out', 'w') as file:
        file.write(f"# simul_mode\t{simul_mode}\n\n")

    # return globals()[simul_mode]() (less verbose but tricky to read)

    if simul_mode == 'charging' :
        p_md, p_sqe, p_batt, p_rce = charging()
    elif simul_mode == 'nyquist' :
        p_md, p_sqe, p_batt, p_rce = nyquist()
    elif simul_mode == 'friction' :
        p_md, p_sqe, p_batt, p_rce = friction()
    else:
        raise ValueError("simul.modus not known")

    return p_md, p_sqe, p_batt, p_rce

def params_init_post(simul_mode, p_md, p_sqe, f_movie = None):

    # init random number generator
    np.random.seed(p_md.random_seed)

    # reset d_time if needed
    d_time = p_sqe.d_time()
    if d_time < p_md.d_time:
        p_md.d_time = d_time
        with open('params.out', 'a') as file:
             if simul_mode == 'friction':
                  file.write(f" # - -  C A U T I O N  ! ! !  - -#")
             file.write(f"# d_time redefined\n")
             file.write(f"d_time\t\t{p_md.d_time:.6g}\n\n")
    else:
        d_time = p_md.d_time

    # random-force pre-factor
    sq_rfp = np.sqrt(6 * p_sqe.resist * p_md.temperature / d_time)

    # prepare directory for movies
    directory = 'Movie'
    if not os.path.exists(directory):
        os.makedirs(directory)
    else:
        files_to_remove = glob.glob(os.path.join(directory, 'config.*.dat'))
        for file in files_to_remove:
            os.remove(file)
        
    with open('params.out', 'a') as file:
        if not ( (f_movie is None) or (f_movie == 0) ):
            file.write("# movies production on\n")
            file.write(f"f_movie\t\t{f_movie:_}\n\n")

        file.write("# - -  derived  parameters  - - #\n\n")

    return d_time, sq_rfp

def make_ion(p_ion):
    r_ion = np.array([0, p_ion.r_y_0, 0])
    v_ion = np.array([0, p_ion.v_y_0, 0])
    force_ion = np.zeros((3,), dtype = float)
    return r_ion, v_ion, force_ion

def make_capacitor(a0, radius, dist):
    r_atom = np.empty((0, 3))
    n_max = int(radius / a0 + 0.5)

    for i_side in range(2):
      for ix in np.arange(-n_max, n_max + 1):
        for iy in np.arange(-n_max, n_max + 1):
          radius_2 = ix**2 + iy**2
          if radius_2 - n_max/4 < n_max**2:
            if i_side == 0 :
              r_atom = np.append(r_atom,[[a0*ix,a0*iy,-dist/2]],axis=0)
            else:
              r_atom = np.append(r_atom,[[a0*ix,a0*iy, dist/2]],axis=0)

    return r_atom

def make_wire(a0, dist, length, wire_height):
    r_atom = np.empty((0, 3))
    n_z = int(length/a0 + 0.5)
    n_x = int(wire_height/a0 + 0.5)
    for i_side in range(2):
      for i_z in range(n_z):
        z = dist/2 + (1 + i_z) * a0
        # four branches parallel to x (inefficient memory alignment)
        if i_side==0 :
          r_atom = np.append(r_atom, [[wire_height, 0, -z]], axis=0)
          r_atom = np.append(r_atom, [[0, 0, -z]], axis=0)
        else:
          r_atom = np.append(r_atom, [[wire_height, 0,  z]], axis=0)
          r_atom = np.append(r_atom, [[0, 0,  z]], axis=0)

      z = dist/2 + n_z * a0
      for i_x in range(0, n_x-1):
        x = (i_x+1) * a0
        if i_side==0:
          r_atom = np.append(r_atom, [[x, 0, -z]], axis=0)
        else:
          r_atom = np.append(r_atom, [[x, 0,  z]], axis=0)

    z = dist/2
    r_atom = np.append(r_atom, [[wire_height, 0, -z]], axis=0)
    r_atom = np.append(r_atom, [[wire_height, 0,  z]], axis=0)

    return(r_atom)

def make_rce(p_rce):
    """
    creates and returns the positions of the atoms populating the RC element
    as well as the indices of the two terminal atoms
    """

    # define position of capacitor atoms
    r_atom = make_capacitor(p_rce.bond_length, p_rce.capa_radius, p_rce.capa_dist)

    # number of capacitor atoms
    n_capa = r_atom.shape[0]//2
    dl_rad = (n_capa/np.pi)**0.5

    # add wire atoms
    r_atom = np.append(r_atom, make_wire(p_rce.bond_length, p_rce.capa_dist,
                       p_rce.wire_length, p_rce.wire_height), axis=0)
    n_atom = r_atom.shape[0]
    n_wire = n_atom // 2 - n_capa - 1

    # dump info to 'params.out'
    params = open('params.out', 'a')
    params.write(f"{dl_rad = :.4g} \t# effective plate radius\n")
    params.write(f"{n_capa = :_} \t# atoms per capacitor plate\n")
    params.write(f"{n_wire = :_} \t# atoms per wire strand (w.o. terminals)\n")
    params.write(f"{n_atom = :_} \t# total number of atoms (w. terminals)\n")
    params.close()

    return r_atom, np.array([n_atom - 2, n_atom - 1]), n_capa, n_wire

@njit
def make_kappa(r, kappa_0):
    n_atom = r.shape[0]
    kappa = np.zeros((n_atom, n_atom))
    for i in range(n_atom):
        kappa[i, i] = kappa_0
        for j in range(i+1, n_atom):
             # Compute the distance between atom i and atom j
             distance = np.linalg.norm(r[i] - r[j])
             kappa[i, j] = 1 / distance 
             kappa[j, i] = kappa[i, j]

    return kappa

@njit  
def make_sq(kappa, r_cut, t_atom, simul_mode):

    idx1 = np.empty((0,), dtype = np.int64) # np.int64 is probably overkill
    idx2 = np.empty((0,), dtype = np.int64)

    n_atom = kappa.shape[0]
    for i_atom in range(n_atom-1):
        for j_atom in range(i_atom+1,n_atom):
             # place split-charges only between nearest neighbors
             if kappa[i_atom, j_atom] >= 1/r_cut:
                 idx1 = np.append(idx1, np.int64(i_atom) )
                 idx2 = np.append(idx2, np.int64(j_atom) )

    # place (internal battery resistance) split-charge between open wire terminals
    idx1 = np.append(idx1, t_atom[0])
    idx2 = np.append(idx2, t_atom[1])

    if 1 > 2: # relict of previous code version
        # Switch off Coulomb interaction btwn terminal atoms (electrolyte screening) 
        kappa[t_atom[0], t_atom[1]] = kappa[t_atom[1], t_atom[0]] = 0
        for i_atom in range(n_atom-2):
            kappa[i_atom, t_atom[0]] = kappa[t_atom[0], i_atom] = 0
            kappa[i_atom, t_atom[1]] = kappa[t_atom[1], i_atom] = 0

    # create arrays with same length as sq_indices
    n_sqe = len(idx1)
    sq_val = np.zeros(n_sqe)
    sq_vel = np.zeros(n_sqe)
    sq_for = np.zeros(n_sqe)
    sq_batt_idx = n_sqe - 1

    return sq_val, sq_vel, sq_for, idx1, idx2, sq_batt_idx

def write_n_sqe(n_sqe):
    params = open('params.out', 'a')
    params.write(f"{n_sqe  = :_} \t# number of split charges\n\n")
    params.close()

@njit
def formal_cr(kappa, n_capa, resist, n_wire):
    q = np.zeros(kappa.shape[0])
    q[:n_capa] = 1 / n_capa
    q[n_capa+1:2*n_capa] = -1 / n_capa
    c_form = 1 / np.dot(q, (kappa @ q) )

    r_form = ( 2 * n_wire + 1 ) * resist
    return(c_form, r_form)

def dump_config(r_atom, charge = None, pot = None, file_name = "config.dat"):
    file = open(file_name, "w")
    n_atom, n_dim = r_atom.shape
    file.write(f"{n_atom}\n")
    for i_atom in range(n_atom):
        if (i_atom < n_atom-2): file.write("\nAg")
        else: file.write("\nK")
        for i_dim in range(n_dim):
            file.write(f"\t{3*r_atom[i_atom, i_dim]:.3g}")
        if charge is not None:
            file.write(f"\t{charge[i_atom]:.3g}")
        if pot is not None:
            file.write(f"\t{pot[i_atom]:.3g}")
    file.close()

def report_cr(c_form, r_form):
    params = open('params.out', 'a')
    params.write("c = " + str(format(c_form, ".6g")) + " \t# formal capacitance\n")
    params.write("r = " + str(format(r_form, ".6g")) + " \t# formal resistance\n")
    params.close()

@njit
def add_sq_2_q(sq_val, sq_idx_1, sq_idx_2, q):
    for idx, sq in enumerate(sq_val):
        q[sq_idx_2[idx]] += sq
        q[sq_idx_1[idx]] -= sq

@njit   
def add_electrostat_pot(q, q_potl, kappa, u, t_atom, coulomb_off = False):
    # apply voltage to battery terminals
    # if u is positive t_atom[0] wants negative electronic charge and is anode,
    # while t_atom[1] is cathode
    q_potl[t_atom[0]] += u/2
    q_potl[t_atom[1]] -= u/2
    # electrostatic + self interaction:
    if coulomb_off: q_potl += np.diag(kappa) * q
    else: q_potl += kappa @ q

@njit   
def add_voltage_2_sq(q_potl, sq_for, sq_idx_1, sq_idx_2, u, sq_batt_idx): 
    for idx, _ in enumerate(sq_for):
        sq_for[idx] += q_potl[ sq_idx_1[idx] ]
        sq_for[idx] -= q_potl[ sq_idx_2[idx] ]
    sq_for[sq_batt_idx] -= u	# change w.r.t. first version :-(

def monitor_charges(moni, md_time, q, n_capa, n_wire, extra = None):
    # monitor q on capacitor, wire, terminals
    # each time first left then right
    q_pl = np.sum(q[:n_capa])
    q_pr = np.sum(q[n_capa:2*n_capa])
    q_wl = np.sum(q[2*n_capa:2*n_capa+n_wire])
    q_wr = np.sum(q[2*n_capa+n_wire:-2])
    q_tot = np.sum(q)
    my_string = f"{md_time:.3f}"
    my_string += f" {q_pl:.4g} {q_pr:.4g} {q_wl:.4g} {q_wr:.4g}"
    # my_string += f" {q[-2]:.4g} {q[-1]:.4g} {q_tot:.4g}"
    # non-standard stuff for trouble shooting goes here
    if extra is not None:
        for z in extra:
            my_string += f" {z:.5g}"
    moni.write(my_string+"\n")
    moni.flush()

@njit
def add_charge_ion_int(q_atom, q_potl, r_atom, force_ion, r_ion):
    # ion is supposed to have a unit charge
    force_ion.fill(0.)
    for idx, _ in enumerate(q_atom):
      r2 = 0    
      dr = r_atom[idx] - r_ion
      dr2 = np.dot(dr, dr)
      dr_abs = np.sqrt(dr2)
      q_potl[idx] += 1./dr_abs
      force_ion += q_atom[idx] * dr / (dr2 * dr_abs)

def report_t_kin(n_t_kin, t_kin_1, t_kin_2, temperature, n_dof):
    t_kin_1 /= n_t_kin
    t_kin_2 /= n_t_kin
    t_kin_2 = (t_kin_2-t_kin_1**2)*n_dof / temperature**2
    t_kin_1 /= temperature
    with open('params.out', 'a') as file:
        file.write("\n# - -  observations  - - #\n\n")
        file.write(f"{t_kin_1 = :.6g}\t# kinetic energy in [k_BT]\n")
        file.write(f"{t_kin_2 = :.6g}\t# specific heat in [k_B]\n")

def main(simul_mode = None, f_movie = False):

    # initialize input parameters 
    p_md, p_sqe, p_batt, p_rce = params_init(simul_mode)
    if simul_mode == 'friction':
        p_ion = ion(p_rce)

    # initialize derived parameters, random number generator, etc.
    d_time, sq_rfp = params_init_post(simul_mode, p_md, p_sqe, f_movie)

    # initialize configuration
    r_atom, t_atom, n_capa, n_wire = make_rce(p_rce)

    if simul_mode == 'friction':
        r_ion, v_ion, force_ion = make_ion(p_ion)

    # define interaction matrix (including self interaction)
    kappa = make_kappa(r_atom, p_sqe.kappa_0)

    # create and initialize split charges
    sq_val, sq_vel, sq_for, sq_idx_1, sq_idx_2, sq_batt_idx = \
    make_sq(kappa, p_sqe.r_cutoff, t_atom, simul_mode)
    write_n_sqe(len(sq_val))

    # initialize atomic charges and potentials
    q_atom = np.zeros(len(r_atom), dtype=float)
    q_potl = q_atom.copy()

    # compute and report formal capacitance and resistance
    c_form, r_form = formal_cr(kappa, n_capa, p_sqe.resist, n_wire)
    report_cr(c_form, r_form)

    # open file(s) for monitoring output
    moni = open("moni.dat", "w")
    if simul_mode == 'friction':
        moni_ion = open("moni_ion.dat", "w")
    else:
        if os.path.exists("moni_ion.dat"):
            os.remove("moni_ion.dat")

    # zero observables 
    n_t_kin, t_kin_1, t_kin_2 = 0, 0, 0
    i_frame = 0

    for i_time in range(-p_md.n_relax, p_md.n_observe):
        md_time = i_time * d_time

        # zero charges, forcesm potentials
        q_atom.fill(0.)
        sq_for.fill(0.)
        q_potl.fill(0.)

        # add resistive + thermal forces
        sq_for -= p_sqe.resist * sq_vel
        if p_md.temperature > 1e-9: 
            sq_for += sq_rfp * ( 2 * np.random.random(sq_for.shape) - 1 )

        # compute charges
        add_sq_2_q(sq_val, sq_idx_1, sq_idx_2, q_atom)

        # compute voltage, electrostatic (self) potential
        u = p_batt.voltage  # * sin( omega * md_time ) # if desired in the future
        add_electrostat_pot(q_atom, q_potl, kappa, u, t_atom)

        # add charge-ion interaction
        if simul_mode == 'friction' :
            add_charge_ion_int(q_atom, q_potl, r_atom, force_ion, r_ion)

        # add voltage/force to sq
        add_voltage_2_sq(q_potl, sq_for, sq_idx_1, sq_idx_2, u, sq_batt_idx)

        # propagate in time ...
        sq_vel += sq_for * d_time / p_sqe.pseudo_l

        # ... with potential no-flow constraint on terminal split charge
        if p_batt.switch_open or (i_time < 0 and simul_mode != 'nyquist') :
            sq_vel[sq_batt_idx] = 0

        sq_val += sq_vel * d_time
        if simul_mode == 'friction' and i_time > 0:
            r_ion += v_ion * d_time

        # monitoring and measurements depend on simulation mode
        f_moni = False
        if simul_mode == 'charging' and (i_time + p_md.n_relax) % 4 == 0:
            f_moni = True
        elif simul_mode == 'nyquist' and i_time > 0 and i_time % 4 == 0:
            f_moni = True
            n_t_kin += 1
            t_kin = np.sum(sq_vel*sq_vel) * p_sqe.pseudo_l / (2*len(sq_vel))
            t_kin_1 += t_kin
            t_kin_2 += t_kin*t_kin
        elif (simul_mode == 'friction') and (i_time % 2 == 0) :
            if i_time + p_md.n_relax % 4 == 0:
                f_moni = True
            if i_time >= 0:
                moni_ion.write(f"{r_ion[1]:.4f} {force_ion[0]:.6g} ")
                moni_ion.write(f"{force_ion[1]:.6g} {md_time:.1f}\n")
        if f_moni:
            monitor_charges(moni, md_time, q_atom, n_capa, n_wire,
            [q_potl[0], q_potl[2*(n_capa+1)], q_potl[-2],
             q_atom[0], q_atom[2*(n_capa+1)], q_atom[-2]])

        if i_time > 0 and f_movie and i_time % f_movie == 0 :
             frame = "Movie/config." + str(i_frame) + ".dat"
             if simul_mode == 'friction':
                 dump_config( np.vstack((r_ion, r_atom)), 
                              np.append(np.array([1]), q_atom), file_name = frame)
             else:
                 dump_config(r_atom, q_atom, file_name = frame)
             i_frame += 1
           
    moni.close()
    if simul_mode == 'friction':
        moni_ion.close()
        dump_config( np.vstack((r_ion, r_atom)), np.append(np.array([1]), q_atom) )
    else:
        dump_config(r_atom, q_atom, q_potl)

    if simul_mode == 'nyquist' and p_md.temperature > 0:
        report_t_kin(n_t_kin, t_kin_1, t_kin_2, p_md.temperature, len(sq_vel))

if __name__ == '__main__':
    # comment in one of the three options
    main('charging', f_movie = 50)
    # main('nyquist', f_movie = 5)
    # main('friction', f_movie = 50)
