
def get_num(file_path=None):
    if file_path is None:
        file_path = './model_name.txt'
    try:
        with open(file_path, 'r') as f:
            line = f.readline()
            try:
                num = int(line)
            except ValueError:
                num = 0
    except FileNotFoundError:
        num = 0
        with open(file_path, 'w') as f:
            f.write(str(num))
    num += 1
    save_num(num)
    return num


def save_num(num, file_path=None):
    if file_path is None:
        file_path = './model_name.txt'
    with open(file_path, 'w') as f:
        f.write(str(num))

