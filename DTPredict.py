def read_model(modelfile):
    with open(modelfile, 'r') as f:
        attline = f.readline().strip()
        attarr = attline.split()
        tokens = f.read().replace('(', ' ( ').replace(')', ' ) ').split()

    root = None
    stack = []
    current_node = None
    index = 0
    while index < len(tokens):
        token = tokens[index]

        if token.startwith('['):
            leaf = Treenode(None)
            leaf.returnval = token[1:-1]

            if stack:
                parent, key = stack.pop()
                leaf.parent = parent
                parent.children[key] = leaf
            else:
                root = leaf
            index += 1
        elif token == '(':
            index += 1
        elif token == ')':
            current_node = None if not stack else stack[-1][0]
            index += 1
        else:
            if current_node is None or current_node.returnval is not None:
                node = Treenode(None)
                node.attribute = token

                if not root:
                    root = node
                current_node = node
                index += 1
            else:
                stack.append((current_node, token))
                index += 1

    return root, attarr

def trace_tree(node, data, attarr):

    while node.returnval is None:
        att = node.attribute
        try:
            index = attarr.index(att) - 1

            if index < 0 or index >= len(data):
                node = list(node.children.values())[0]
            else:
                val = data[index]
                node = node.children.get(val, list(node.children.values())[0])
            except ValueError:
                node = list(node.children.values())[0]
        return node.returnval

def DTpredict(data, model, prediction):
    global root, atts
    root, atts = read_model(model)

    predictions = []
    with open(data, 'r') as f:
        for line in f:
            tokens = line.strip().split()
            datapoint = tokens[1:]
            pred = trace_tree(root, datapoint,atts)
            predictions.append(pred)

    with open(prediction, 'w') as f:
        for pred in predictions:
            f.write(pred + '\n')