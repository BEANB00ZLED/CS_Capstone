def first_value_loc(list_of_lists: list, index: int):
    if len(list_of_lists) > 0:
        while isinstance(list_of_lists[0], list):
            list_of_lists = list_of_lists[0]
        return list_of_lists[index]
    else:
        return None