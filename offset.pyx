from cpython cimport array
import array
import numpy as np

cpdef find_offset(pquery_hps, pref1_hps):
    
    cdef array.array qa = array.array('Q', pquery_hps)
    cdef unsigned long long[:] query_hps = qa
    cdef array.array ra1 = array.array('Q', pref1_hps)
    cdef unsigned long long[:] ref1_hps = ra1
    
    cdef unsigned long len1, lenq, offset_range, offset, min_score1, min_offset1, cumm_dist1, idx, max_score, curr_score, curr_max
    cdef unsigned long long i, hp1, rf1
    cdef double tamper_score

    len1 = len(ref1_hps)
    lenq = len(query_hps)
    
    offset_range = len1 - lenq + 1

    min_score1 = 64 * offset_range
    min_offset1 = 0

    
    for offset in range(offset_range):
        cumm_dist1 = 0
        
        for idx in range(lenq):
            hp1 = query_hps[idx]
            rf1 = ref1_hps[offset + idx]
            
            
            i = hp1 ^ rf1
            i = i - ((i >> 1) & <unsigned long long> 0x5555555555555555)
            i = (i & <unsigned long long> 0x3333333333333333) + ((i >> 2) & <unsigned long long> 0x3333333333333333)
            cumm_dist1 += (((i + (i >> 4)) & <unsigned long long> 0xF0F0F0F0F0F0F0F) * <unsigned long long> 0x101010101010101) >> 56


        if cumm_dist1 < min_score1:
            min_offset1 = offset
            min_score1 = cumm_dist1
    
    return min_offset1