def get_pseudocode1():
    with open(f'pseudocode1.txt', 'r') as f:
        return f.read()

def get_pseudocode2():
    with open(f'pseudocode2.txt', 'r') as f:
        return f.read()

def get_trr():
    with open(f'prompts/trr_solve_framework.txt', 'r') as f:
        return f.read()

def get_problem():
    with open(f'coding_problem.txt', 'r') as f:
        return f.read()
        