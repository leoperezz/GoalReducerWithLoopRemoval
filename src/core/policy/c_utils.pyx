import numpy as np
cimport numpy as np

def say_hello_to(name):
    print("Hello again %s!" % name)

def remove_loops(long[:] a):
    cdef Py_ssize_t length_a = a.shape[0]
    cdef list[long] clean_a = []
    cdef list[long] clean_a_ids = []
    for aid in range(length_a):
        a_ele = a[aid]

        if a_ele in clean_a:
            a_ele_idx = clean_a.index(a_ele)
            clean_a = clean_a[:a_ele_idx + 1]
            clean_a_ids = clean_a_ids[:a_ele_idx + 1]
        else:
            clean_a.append(a_ele)
            clean_a_ids.append(aid)

    return clean_a, clean_a_ids


def trj_worker(long[:] indices_single):
    cdef list[long] trj_ids_ids
    _, trj_ids_ids = remove_loops(indices_single)
    assert trj_ids_ids[0] == 0

    cdef float trj_w
    cdef long subgoal_index
    cdef long terminal_index

    if len(trj_ids_ids) < 2:
        trj_w = 0.0
        subgoal_index = indices_single[0]
        terminal_index = indices_single[0]
    else:
        noloop_length = len(trj_ids_ids)

        trj_w = 1.0 / (noloop_length + 5.0)
        id_nids = np.arange(noloop_length)
        terminal_idid = np.random.choice(id_nids[1:], 1).item()  # at least 2
        if len(id_nids[1:terminal_idid]) == 0:
            sg_idid = terminal_idid
        else:
            sg_idid = np.random.choice(id_nids[1 : terminal_idid + 1], 1).item()

        terminal_index = indices_single[terminal_idid]
        subgoal_index = indices_single[sg_idid]

    return trj_w, subgoal_index, terminal_index


def gen_ind_res(int bsz, long[:,:] indicess):
    res = []
    for idx_in_batch in range(bsz):
        res.append(trj_worker(indicess[:, idx_in_batch]))
    return res

