from queue import Queue
from tensorflow import keras
import logging

def nodes_by_length(model):
    return model._nodes_by_length

def nodes_in_order(model):
    nodes_by_depth = model._nodes_by_depth
    depth_keys = list(nodes_by_depth.keys())
    depth_keys.sort(reverse=True)

    for depth in depth_keys:
      nodes = nodes_by_depth[depth]
      for node in nodes:
          yield node

def module_map(model: keras.Model, filter, mapper):
    """
    mapper takes argument: (inputs, old_layer) and returns new desired output
    """
    new_tensors = {}
    
    for node in nodes_in_order(model):
        # Generate the new output for node node
        if len(node.parent_nodes) == 0:
            new_outputs = node.outputs
        else:
            new_inputs = []
            for old_input in node.keras_inputs:
                new_inputs.append(new_tensors[str(id(old_input))])

            layer = node.layer
            if len(new_inputs) == 1:
                new_inputs = new_inputs[0]
            if filter(layer):
                new_outputs = mapper(new_inputs, layer)
            else:
                new_outputs = node.layer(new_inputs)
        if not isinstance(new_outputs, list):
            new_outputs = [new_outputs]
        for old, new in zip(node.flat_output_ids, new_outputs):
            new_tensors[old] = new

    new_inputs = model.inputs
    new_outputs = []
    for old_output in model.outputs:
        new_output = new_tensors[str(id(old_output))]
        new_outputs.append(new_output)
    return keras.Model(new_inputs, new_outputs, name=model.name)

def insert_layer_nonseq(model, condition, insert_layer_factory=None, position='after'):
    """
    Insert a layer before, after or replacing the layers that matches the conditions, 
        or delete the layers
    
        - model: the Keras Functional model to operate on; function generates a new model
        - condition: f(layer) -> bool as the filter function
        - insert_layer_factory: f(old_layer) -> new_layer
        - position: {before, after, delete, replace}
    """
    def mapper_func(layer_input, layer):
        if condition(layer):
            if position == 'replace' or position == 'before':
                x = layer_input
            elif position == 'after':
                x = layer(layer_input)
            elif position == 'delete':
                pass
            else:
                raise ValueError('position must be: before, after, delete or replace')

            if position != 'delete':
                new_layer = insert_layer_factory(layer)
                x = new_layer(x)
                logging.debug('New layer: {} Old layer: {} Type: {}'.format(new_layer.name,
                                                                layer.name, position))
                if position == 'before':
                    x = layer(x)
            else:
                x = layer_input
                logging.debug('Deleted layer: {}'.format(layer.name))
        else:
            x = layer(layer_input)
        return x
    return module_map(model, condition, mapper_func)
