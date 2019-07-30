def print_dict(dict_name):
    for key in dict_name.__dict__:
        print(key, '=', dict_name.__dict__[key])
