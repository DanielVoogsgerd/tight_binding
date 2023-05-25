import unittest

import tight_binding.model
from tight_binding.objects import UnitCell, Atom
from tight_binding.unit_cells import load_unit_cell
from sympy.vector import CoordSys3D


class TestSinglePerovskite(unittest.TestCase):
    unit_cell: UnitCell

    def setUp(self):
        self.r = CoordSys3D('r')
        r = self.r

        self.unit_cell = load_unit_cell("resources/unit_cells/CsPbI3.json", r)

    def test_find_neighbours(self):
        distance, neighbours_dict, neighbours_set = self.unit_cell.find_neighbours()

        self.assertEqual(len(neighbours_dict), 4)

        self.assertTrue("Pb" in neighbours_dict[self.unit_cell.get_atom(0)])
        self.assertEqual(len(neighbours_dict[self.unit_cell.get_atom(0)]), 1)
        self.assertEqual(len(neighbours_dict[self.unit_cell.get_atom(0)]["Pb"]), 6)

        self.assertTrue("I" in neighbours_dict[self.unit_cell.get_atom(1)])
        self.assertEqual(len(neighbours_dict[self.unit_cell.get_atom(1)]), 1)
        self.assertEqual(len(neighbours_dict[self.unit_cell.get_atom(1)]["I"]), 2)

        self.assertTrue("I" in neighbours_dict[self.unit_cell.get_atom(2)])
        self.assertEqual(len(neighbours_dict[self.unit_cell.get_atom(2)]), 1)
        self.assertEqual(len(neighbours_dict[self.unit_cell.get_atom(2)]["I"]), 2)

        self.assertTrue("I" in neighbours_dict[self.unit_cell.get_atom(3)])
        self.assertEqual(len(neighbours_dict[self.unit_cell.get_atom(3)]), 1)
        self.assertEqual(len(neighbours_dict[self.unit_cell.get_atom(3)]["I"]), 2)

        self.assertEqual(neighbours_set, {('Pb', 'I'), ('I', 'Pb')})

    def test_matrix(self):
        model = tight_binding.model.TightBinding(self.unit_cell, self.r)
        matrix = model.construct_matrix()
        self.assertTrue(matrix.is_hermitian)

    def test_Gamma(self):
        pass
