import sys
import math

def lcm(x, y):
    return abs(x * y) // math.gcd(x, y)

def dfs(node, tree, values, parent, subtree_lcm):
    # Initialize LCM of the current subtree with the node's value
    current_lcm = values[node]
    
    for child in tree[node]:
        if child != parent:  # Avoid revisiting the parent node
            current_lcm = lcm(current_lcm, dfs(child, tree, values, node, subtree_lcm))
    
    subtree_lcm[node] = current_lcm
    return current_lcm

def get_answer(n, values, parent):
    # Build the tree using adjacency list
    tree = [[] for _ in range(n)]
    for i in range(n - 1):
        tree[parent[i]].append(i + 1)  # Parent of node i+1 is parent[i]
        tree[i + 1].append(parent[i])
    
    # Initialize subtree LCM array
    subtree_lcm = [-1] * n
    
    # Compute LCM of all subtrees
    dfs(0, tree, values, -1, subtree_lcm)
    
    # Total LCM of the entire tree
    total_lcm = subtree_lcm[0]
    
    # Count good subtrees
    good_subtrees_count = 0
    for i in range(n):
        remaining_lcm = total_lcm // subtree_lcm[i]  # LCM of the rest of the tree
        if subtree_lcm[i] == remaining_lcm:
            good_subtrees_count += 1
    
    return good_subtrees_count

def main():
    # Reading input
    n = int(input("Enter number of nodes: ").strip())
    print(f"Number of nodes: {n}")

    # Read node values
    print(f"Enter {n} values for the nodes:")
    values = [int(input().strip()) for _ in range(n)]
    print(f"Node values: {values}")
    
    # Read parent array (n-1 edges for n nodes)
    print(f"Enter {n-1} parent indices (0-based):")
    parent = [int(input().strip()) for _ in range(n - 1)]
    print(f"Parent array: {parent}")
    
    # Call the function and print the result
    result = get_answer(n, values, parent)
    print(f"Output: {result}")

if __name__ == "__main__":
    main()


