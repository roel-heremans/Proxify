def get_alternating_values(maxima_indices, minima_indices):
    '''
    This function serves to get alternating values between maxima_indices and minima_indices. For instance:
    maxima_indices = [13, 22, 24, 38]
    minima_indices = [15, 27, 42, 53]
    the result would be [13, 15, 22, 27, 38, 42]
    :param maxima_indices: list that contains the indices where the maxima are located
    :param minima_indices: list that contians the indices where the minma are located
    :return: list that contains the alternating indices between max and minima or between min and maxima
    '''
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
            successive_indices.append(i + 1)

    # indices would not correspond anymore to the one mentioned by successive _indices when doing this operation from
    # the beginning to the end, need to do this from last to first, hence [::-1]
    for i in successive_indices[::-1]:
        removed = merged_indices_res.pop(i)
        # print('removed idx: {}, value: {}'.format(i, removed))

    if merged_indices_res[0] == maxima_indices[0]:
        max_is_first = 1
    else:
        max_is_first = 0

    # check if the last is a maximum:  if so remove it from the list
    if max_is_first:
        if merged_indices_res[-1] in maxima_indices:
            merged_indices_res.pop()
    else:
        if merged_indices_res[-1] in minima_indices:
            merged_indices_res.pop()

    return merged_indices_res, max_is_first


if __name__ == "__main__":

    # example with consecutive minima without a maxima in between, there are maxima at the end that needs to be dropped
    maxima_indices = [177, 195, 226, 243, 275,           287, 299, 318, 328, 348, 380, 403, 415, 433, 500, 511, 523, 533, 567, 590, 617]
    minima_indices = [  182, 217, 237, 257, 260, 270, 283, 293, 304, 323, 334, 357, 386, 411, 426, 462, 506, 517, 528, 549, 572]
    result = [177,182, 195, 217, 226, 237, 243, 257, 275, 283, 287, 293, 299, 304, 318, 323, 328, 334, 348, 357, 380,
              386, 403, 411, 415, 426, 433, 462, 500, 506, 511, 517, 523, 528, 533, 549, 567, 572]
    my_dict = {'01': {'max_list': maxima_indices, 'min_list': minima_indices, 'result': result} }

    maxima_indices = [13, 21,         32, 42, 55]
    minima_indices = [  17, 18, 19, 25          ]
    result = [13, 17, 21, 25]
    my_dict = {'02': {'max_list': maxima_indices, 'min_list': minima_indices, 'result': result} }

    #example with consecutive minima without intelacing maxima, as well as consecutive maxima without interlacing minima
    maxima_indices = [          21, 32, 42, 55  ]
    minima_indices = [17, 18, 19, 25,         60]
    result = [17, 21, 25, 32]
    my_dict = {'03': {'max_list': maxima_indices, 'min_list': minima_indices, 'result': result} }

    for key in my_dict.keys():
        alternating_extrema, max_is_first = get_alternating_values(my_dict[key]['max_list'], my_dict[key]['min_list'])
        assert my_dict[key]['result'] == alternating_extrema, "get_alternating_values has problem with example {}".format(key)


