

def merge_indices(maxima_indices, minima_indices):
    merged_indices_res = []
    source_info = []

    max_idx = 0
    min_idx = 0

    while max_idx < len(maxima_indices) and min_idx < len(minima_indices):
        max_val = maxima_indices[max_idx]
        min_val = minima_indices[min_idx]

        if min_val <= max_val:
            merged_indices_res.append(min_val)
            source_info.append('min')
            min_idx += 1
        else:
            merged_indices_res.append(max_val)
            source_info.append('max')
            max_idx += 1

    # Add the remaining elements from either list
    merged_indices_res.extend(maxima_indices[max_idx:])
    source_info.extend(['max'] * (len(maxima_indices) - max_idx))

    merged_indices_res.extend(minima_indices[min_idx:])
    source_info.extend(['min'] * (len(minima_indices) - min_idx))

    return merged_indices_res, source_info

def find_successive_values(source_info):
    successive_indices = []

    # Loop through the indices to find successive elements
    for i in range(len(source_info) - 1):
        if source_info[i] == source_info[i + 1]:
            successive_indices.append(i+1)
    return successive_indices

def get_alternating_values(maxima_indices, minima_indices):
    merged_indices_res = []
    source_info = []

    max_idx = 0
    min_idx = 0

    while max_idx < len(maxima_indices) and min_idx < len(minima_indices):
        max_val = maxima_indices[max_idx]
        min_val = minima_indices[min_idx]

        if min_val <= max_val:
            merged_indices_res.append(min_val)
            source_info.append('min')
            min_idx += 1
        else:
            merged_indices_res.append(max_val)
            source_info.append('max')
            max_idx += 1

    # Add the remaining elements from either list
    merged_indices_res.extend(maxima_indices[max_idx:])
    source_info.extend(['max'] * (len(maxima_indices) - max_idx))

    merged_indices_res.extend(minima_indices[min_idx:])
    source_info.extend(['min'] * (len(minima_indices) - min_idx))

    successive_indices = []

    # Loop through the indices to find successive elements
    for i in range(len(source_info) - 1):
        if source_info[i] == source_info[i + 1]:
            successive_indices.append(i+1)

    for i in successive_indices[::-1]:
        removed = merged_indices_res.pop(i)
        print('removed idx: {}, value: {}'.format(i, removed))

    if merged_indices_res[0] == maxima_indices[0]:
        max_is_first = 1
    else:
        max_is_first = 0

    return merged_indices_res, max_is_first

if __name__ == "__main__":
    maxima_indices = [177, 195, 226, 243, 275, 287, 299, 318, 328, 348, 380, 403, 415, 433, 500, 511, 523, 533, 567, 590, 617]
    minima_indices = [182, 217, 237, 257, 260, 270, 283, 293, 304, 323, 334, 357, 386, 411, 426, 462, 506, 517, 528, 549, 572]
    alternating_extrema, max_is_first = get_alternating_values(maxima_indices, minima_indices)


    a=1