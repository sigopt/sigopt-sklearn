import random


def random_assignments(experiment_definition):
  def rand_param(param):
    if param['type'] == 'int':
      return random.randint(param['bounds']['min'], param['bounds']['max'])
    elif param['type'] == 'double':
      return random.uniform(param['bounds']['min'], param['bounds']['max'])
    elif param['type'] == 'categorical':
      return random.choice([c['name'] for c in param['categorical_values']])
    else:
      raise Exception('Unknown parameter type {}!'.format(param.type))

  return {p['name']: rand_param(p) for p in experiment_definition['parameters']}
