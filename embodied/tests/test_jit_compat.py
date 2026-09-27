import ast
import pathlib
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[2]


class TestJitCalls(unittest.TestCase):
  def test_no_positional_jit_options(self):
    for path in ('embodied/jax/transform.py', 'embodied/jax/agent.py',
                 'embodied/jax/internal.py'):
      tree = ast.parse((ROOT / path).read_text())
      for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == 'jax' and node.func.attr == 'jit'):
          self.assertLessEqual(len(node.args), 1, f'{path}:{node.lineno}')
