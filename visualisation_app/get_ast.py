import ast


def get_ast(filename:str):
    with open(filename,"r") as f:
        src_file=f.read()

    tree=ast.parse(src_file)
    return tree


if __name__ == '__main__':
    tree=get_ast("../demos/exponential_decay.py")
    print(tree)
    for node in ast.walk(tree):
        print(node)