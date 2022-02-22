import numpy as np
import multiprocessing as ml
from linckage import msprime_simulate_variants
import time

class Variant:
    def __init__(self, genotype):
        self.genotypes = genotype


def internal_incompatibility(genotype1, genotype2, oriented=True):
    A = {i for i in range(len(genotype1)) if genotype1[i]}
    B = {i for i in range(len(genotype2)) if genotype2[i]}
    return not (not A.intersection(B) or A.union(B) == B or A.union(B) == A)


def external_incompatibility(segregative_site, variants, revert=False):
    segregated, position = segregative_site
    if revert:
        position = len(variants) - position
    for index, variant in enumerate(variants[position:]):
        if len(set(variant.genotypes[segregated])) == 2 \
        and 1 in variant.genotypes[np.invert(segregated)]:
            return  len(variants[position:]) - index - 1 if revert else index + position
    # return  0 if revert else len(variants) - 1
    return -1


def shortening(list_incompatible):
    list_incompatible = sorted(list_incompatible, key=lambda t: (t[0], t[1]))
    index = len(list_incompatible) - 1
    while index > 0:
        # print(list_incompatible)
        min1, max1 = list_incompatible[index - 1]
        min2, max2 = list_incompatible[index]
        if min1 <= min2 and max1 >= max2:
            list_incompatible[index - 1] = list_incompatible[index]
            del list_incompatible[index]
        elif min1 >= min2 and max1 <= max2:
            del list_incompatible[index]
        elif min1 < min2 and max1 <= max2 and max1 > min2:
            del list_incompatible[index]
            list_incompatible[index - 1] = (min2, max1)
        # elif min2 <= min1 and min2 >= max2 and max2 <= max1:
        #     del list_incompatible[index]
        #     list_incompatible[index - 1] = (min1, max2)
        index -= 1
    return list_incompatible


def segregative_sites(variants):
    list_segregative_sites = []
    for index, variant in enumerate(variants):
        tmp = sum(variant.genotypes)
        if tmp > 2 and tmp < len(variant.genotypes) :
            list_segregative_sites.append((list(variant.genotypes == 1), index))
    return list_segregative_sites


def detect_internal_incompatibilities(variants, segregated, borne_inf, borne_sup, thresold=150):
    list_incompatible_sites = []
    for i in range(borne_inf, borne_sup + 1):
        j = i + 1
        while j < i + thresold and j < borne_sup + 1:
            if internal_incompatibility(variants[i].genotypes[segregated],
            variants[j].genotypes[segregated]):
                list_incompatible_sites.append((i, j))
            j += 1
    return list_incompatible_sites


def non_ovorlaping_incompatibilities(variants, segregative_site):
    segregated, position_segragative_site = segregative_site
    borne_sup = external_incompatibility(segregative_site, variants, revert=False)
    borne_inf = external_incompatibility(segregative_site, variants[::-1], revert=True)
    borne_inf = 0 if borne_inf == -1 else borne_inf
    borne_sup = len(variants) - 1 if borne_sup == -1 else borne_sup
    list_incompatible_sites = detect_internal_incompatibilities(variants, segregated,
    borne_inf, borne_sup)
    list_incompatible_sites.append((borne_inf, position_segragative_site))
    list_incompatible_sites.append((position_segragative_site, borne_sup))
    return shortening(list_incompatible_sites)


def detect_mld_block(variants, segregated_site):
    segregated, position_segragative_site = segregated_site
    list_incompatible_sites = non_ovorlaping_incompatibilities(variants, segregated_site)
    for index in range(len(list_incompatible_sites) - 1):
        borne_sup_inf = list_incompatible_sites[index][1]
        borne_inf_sup = list_incompatible_sites[index + 1][0]
        if borne_sup_inf <= position_segragative_site \
        and borne_inf_sup >= position_segragative_site:
            return (list_incompatible_sites[index], \
            list_incompatible_sites[index + 1], \
            segregated)
    return((0, 0), (len(variants) - 1, len(variants) -1), segregated)


def mld_blocks_for_all_segregative_sites(variants):
    list_segregative_sites = segregative_sites(variants)
    pool = ml.Pool(4)
    data = pool.starmap(detect_mld_block, [(variants, segregated) for segregated in list_segregative_sites])
    pool.close()
    return data


def main():
    start = time.time()
    params = {"sample_size": 8, "Ne": 1, "ro": 8e-4, "mu": 8e-2,  "Tau": 1.0,
    "Kappa": 1.0 , "length": int(1e5)}
    variants = list(msprime_simulate_variants(params).variants())
    print(len(variants))
    print(mld_blocks_for_all_segregative_sites(variants))
    print(time.time() - start)

if __name__ == "__main__":
    main()
