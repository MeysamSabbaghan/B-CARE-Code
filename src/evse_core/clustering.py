"""EVSE clustering + frame/mass construction (adapted from official EVSE repo).

Faithful to Algorithm 1 / Steps 2–4 of Kunitomo-Jacquin et al. (EACL 2026).
"""

from __future__ import annotations

import copy
import logging
from itertools import chain
from typing import Any

import networkx as nx
import numpy as np

from .evidence import my_bin2dec, vector_to_set_expression


def all_entailments(strings_list: list[str], check_implication) -> list[list[int]]:
    """Pairwise NLI matrix. Value 2 == entailment (Deberta MNLI argmax convention)."""
    n = len(strings_list)
    matrix = [[-1] * n for _ in range(n)]
    for i, string1 in enumerate(strings_list):
        for j, string2 in enumerate(strings_list):
            if string1 == string2:
                matrix[i][j] = 2
            else:
                pred, _ = check_implication(string1, string2)
                matrix[i][j] = int(pred)
    return matrix


def get_semantic_ids_and_hierarchy(strings_list: list[str], entailments_full_matrix: list[list[int]]) -> dict:
    n = len(strings_list)
    mprim = np.array(entailments_full_matrix, dtype=int).copy()
    semantic_set_ids = list(range(n))
    nprim = n

    while True:
        relations = [
            (i, j)
            for i, row in enumerate(mprim)
            for j, value in enumerate(row)
            if value == 2 and i != j
        ]
        g = nx.DiGraph(relations)
        try:
            cycle = nx.find_cycle(g, orientation="original")
        except nx.NetworkXNoCycle:
            break
        i, j = cycle[0][0], cycle[0][1]
        for k in range(n):
            if semantic_set_ids[k] == j:
                semantic_set_ids[k] = i
        for k in range(nprim):
            if mprim[i][k] != mprim[j][k]:
                mprim[i][k] = -1
            if mprim[k][i] != mprim[k][j]:
                mprim[k][i] = -1
        mprim = np.delete(mprim, j, axis=0)
        mprim = np.delete(mprim, j, axis=1)
        for k in range(n):
            if semantic_set_ids[k] > j:
                semantic_set_ids[k] = semantic_set_ids[k] - 1
        nprim = mprim.shape[0]

    r_list = []
    for i in range(nprim):
        for j in range(nprim):
            if i != j and mprim[i][j] == 2:
                r_list.append([i, j])
    return {"c": semantic_set_ids, "R": r_list}


def eliminate_redundant_relations(edges: list[list[int]]) -> list[list[int]]:
    g = nx.DiGraph()
    g.add_edges_from(edges)
    non_redundant = []
    for u, v in list(g.edges()):
        g.remove_edge(u, v)
        if not nx.has_path(g, u, v):
            non_redundant.append([u, v])
        g.add_edge(u, v)
    return non_redundant


def build_hierarchy(entailments_full_matrix: list[list[int]], unique_responses: list[str]) -> dict:
    res = get_semantic_ids_and_hierarchy(unique_responses, entailments_full_matrix)
    relations = copy.deepcopy(res["R"])
    cluster_belonging = copy.deepcopy(res["c"])
    gtemp = nx.DiGraph(relations)
    relations_add_single_root = copy.deepcopy(relations)

    clusters_list = list(set(res["c"]))
    strings_in_rel = list(set(chain.from_iterable(res["R"]))) if res["R"] else []
    root_nodes = [node for node, out_deg in gtemp.out_degree() if out_deg == 0] + [
        k for k in clusters_list if k not in strings_in_rel
    ]
    for root in root_nodes:
        relations_add_single_root.append([root, -1])

    cluster2names: dict[str, str] = {}
    for cluster in clusters_list:
        temp = [unique_responses[i] for i, value in enumerate(cluster_belonging) if value == cluster]
        cluster2names[str(cluster)] = "\n or \n".join(temp)

    relations = eliminate_redundant_relations(relations_add_single_root)
    hierarchy = copy.deepcopy(relations)
    parents: list[int] = []
    for r in hierarchy:
        if r[1] not in parents:
            parents.append(r[1])

    is_a_closed = hierarchy[:]
    nodes2names = copy.deepcopy(cluster2names)
    nodes2names["-1"] = "Ignorance"

    k = 1
    max_c = max(cluster_belonging) if cluster_belonging else 0
    for parent in parents:
        if parent == -1:
            continue
        child_count = 0
        single_child = None
        for r in hierarchy:
            if r[1] == parent:
                child_count += 1
                if child_count > 1:
                    break
                single_child = r[0]
        if child_count != 1:
            continue
        fill_id = max_c + k
        is_a_closed.append([fill_id, parent])
        filling_node = f"{nodes2names[str(parent)]}\n but not \n{nodes2names[str(single_child)]}"
        nodes2names[str(fill_id)] = filling_node
        k += 1

    fill_id = max_c + k
    is_a_closed.append([fill_id, -1])
    nodes2names[str(fill_id)] = "semantically unrelated tail"

    return {
        "relation": is_a_closed,
        "cluster_belonging": cluster_belonging,
        "unique_responses": unique_responses,
        "nodes2names": nodes2names,
    }


def build_frame_of_discernment(hierarchy_info: dict) -> dict:
    discernment_info: dict[str, Any] = {}
    c_list = list(hierarchy_info["nodes2names"].keys())
    discernment_info["C"] = c_list

    omega = []
    for c in c_list:
        c_is_el = True
        for rel in hierarchy_info["relation"]:
            if str(rel[1]) == c:
                c_is_el = False
                break
        if c_is_el:
            omega.append(c)
    discernment_info["Omega"] = omega

    c_bin = np.zeros((len(omega), len(c_list)), dtype=int)
    totreat = c_list[:]
    for i in range(len(omega)):
        c_bin[i, c_list.index(omega[i])] = 1
        totreat.remove(omega[i])

    k = 0
    while totreat:
        candidate = totreat[k]
        treat = True
        childs = []
        for r in hierarchy_info["relation"]:
            if str(r[1]) == candidate:
                if str(r[0]) in totreat:
                    treat = False
                    break
                childs.append(str(r[0]))
        if treat:
            k = 0
            totreat.remove(candidate)
            for child in childs:
                indices = [i for i, x in enumerate(c_bin[:, c_list.index(child)].tolist()) if x == 1]
                for index in indices:
                    c_bin[index, c_list.index(candidate)] = 1
        else:
            k += 1

    discernment_info["C_bin"] = np.array(c_bin)
    c_dec = [my_bin2dec(list(c_bin[:, i])) for i in range(len(c_list))]
    discernment_info["C_dec"] = c_dec
    discernment_info["Omega_dec"] = max(c_dec) if c_dec else 0
    discernment_info["Omega_size"] = int(c_bin.shape[0])
    discernment_info["C_dec2C_bin"] = {c_dec[i]: c_bin[:, i] for i in range(len(c_list))}
    discernment_info["C_dec2set_name"] = {
        c_dec[i]: vector_to_set_expression(list(c_bin[:, i])) for i in range(len(c_list))
    }
    discernment_info["C_key_to_dec"] = {c_list[i]: c_dec[i] for i in range(len(c_list))}
    return discernment_info


def mass_assignment(
    log_liks_with_eos: list[list[float]],
    responses: list[str],
    hierarchy_info: dict,
    discernment_info: dict,
) -> dict:
    """Eqs. 6–7. Unique answers only; no length-norm; EOS included in provided logliks."""
    sequence_logprob = [float(np.sum(log_lik)) for log_lik in log_liks_with_eos]
    sequence_prob = [float(np.exp(lp)) for lp in sequence_logprob]
    unique = hierarchy_info["unique_responses"]
    sequence_prob_unique = [sequence_prob[responses.index(response)] for response in unique]

    if sum(sequence_prob_unique) > 1.0:
        s = sum(sequence_prob_unique)
        sequence_prob_unique = [p / s for p in sequence_prob_unique]
        logging.warning("sum sequence_prob_uniqueEvents exceeded 1; renormalized.")

    cluster_belonging = hierarchy_info["cluster_belonging"]
    cluster_list = sorted(set(cluster_belonging))
    proba_by_cluster: dict[int, float] = {c: 0.0 for c in cluster_list}
    for event_index, cluster_id in enumerate(cluster_belonging):
        proba_by_cluster[cluster_id] += sequence_prob_unique[event_index]

    f_list: list[int] = []
    m_list: list[float] = []
    f_names: list[str] = []
    key_to_dec = discernment_info["C_key_to_dec"]
    for focal_cluster in cluster_list:
        f_list.append(int(key_to_dec[str(focal_cluster)]))
        m_list.append(float(proba_by_cluster[focal_cluster]))
        f_names.append(hierarchy_info["nodes2names"][str(focal_cluster)])

    mass_obs = sum(m_list)
    if mass_obs < 0.999:
        f_list.append(int(discernment_info["Omega_dec"]))
        m_list.append(float(1.0 - mass_obs))
        f_names.append(hierarchy_info["nodes2names"]["-1"])

    return {
        "F": f_list,
        "M": m_list,
        "F_names": f_names,
        "m_theta": float(1.0 - mass_obs) if mass_obs < 0.999 else 0.0,
        "observed_mass": float(min(mass_obs, 1.0)),
        "unique_sequence_probs": {
            unique[i]: sequence_prob_unique[i] for i in range(len(unique))
        },
    }
