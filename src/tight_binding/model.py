#!/usr/bin/env python

from sympy import (
    exp,
    symbols,
    I,
    Mul,
    zeros,
    diag,
    Matrix,
    Expr,
    sqrt,
    Symbol,
    lambdify,
)
from sympy.vector import CoordSys3D, dot, BaseVector
from tight_binding.objects import Orbital, UnitCell, Atom, Neighbour, Spin
from tight_binding.slaterkoster import ParamCollection
import itertools
import sys
import numpy.linalg as LA
import numpy as np
import numpy.typing as npt

from typing import Set, Dict, Tuple, List, Optional, Union, Self, Callable

from tight_binding.spin_orbit import get_soc_matrix

np.set_printoptions(threshold=sys.maxsize)

ORBITAL_CLASSES = ["s", "p", "d"]

ORBITAL_MAP = {
    "s": ["s"],
    "p": ["px", "py", "pz"],
    "d": ["xz", "yz", "xy", "x2-y2", "3z2-r2"],
}

Number = Union[float, int]


class Basis:
    def __init__(self) -> None:
        self.basis: List[Tuple[Atom, Orbital, Optional[Spin]]] = []

    @classmethod
    def from_unit_cell(cls, unit_cell: UnitCell) -> Self:
        basis = cls()
        atoms = unit_cell.atoms
        for atom in atoms:
            for orbital_class, orbitals in atom.active_orbitals.items():
                basis.basis.extend(
                    itertools.product(
                        [atom],
                        orbitals,
                        tuple(Spin) if unit_cell.spin_orbit_coupling else (None,),
                    )
                )

        return basis

    def get_index(self, atom: Atom, orbital: Orbital, spin: Optional[Spin]) -> int:
        return self.basis.index((atom, orbital, spin))


class TightBinding:
    overlap_integrals: Dict[Tuple[str, str], ParamCollection]
    energies: Dict[Tuple[str, str], Symbol]
    unit_cell: UnitCell

    def __init__(
        self, unit_cell: UnitCell, r_symbol: CoordSys3D, spin_orbit: bool = False
    ) -> None:
        self.unit_cell = unit_cell
        self.basis = Basis.from_unit_cell(unit_cell)

        self.r = r_symbol

        self.coord_symbols = symbols("x y z", real=True)
        self.angle_symbols = symbols("l m n", real=True)
        self.k_symbols = symbols("kx ky kz", real=True)
        self.k_vector = sequence_to_sympy_vector(self.r, self.k_symbols)

        self.energy_symbols: Set[Symbol] = set()
        self.energy_integral_symbols: Set[Symbol] = set()
        self.soc_symbols: Set[Symbol] = set()

    def construct_hamiltonian(self, n: int) -> Matrix:
        if n != 1:
            raise NotImplementedError(
                "Creating a n nearest neighbour hamiltionian"
                "n > 1 is not yet supported."
            )

        all_neighbours = self.unit_cell.sort_neighbours()

        matrix = zeros(len(self.basis.basis))

        # TODO: Support n nearest neighbours instead of one.
        distance, neighbours = all_neighbours[0]

        matrix += self._get_tb_matrix(neighbours)

        if self.unit_cell.spin_orbit_coupling:
            matrix += self._get_soc_matrix()

        return matrix

    def _get_tb_matrix(self, neighbours: Dict[Atom, List[Neighbour]]) -> Matrix:
        matrix = zeros(len(self.basis.basis))

        # Diagonal first
        for i, (atom, orbital, spin) in enumerate(self.basis.basis):
            energy_symbol = self.get_energy_symbol(atom, orbital)
            self.energy_symbols.add(energy_symbol)
            matrix[i, i] = energy_symbol

        # Fill in all the effect of all neighbouring atoms
        for j, (ket_atom, ket_orbital, ket_spin) in enumerate(self.basis.basis):
            for neighbour in neighbours[ket_atom]:
                for orbital_class, orbitals in neighbour.atom.active_orbitals.items():
                    for orbital in orbitals:
                        i = self.basis.get_index(neighbour.atom, orbital, ket_spin)
                        relative_vector = (
                            neighbour.observed_coordinate - ket_atom.coordinate
                        )
                        bloch_factor = calculate_bloch_factor(
                            relative_vector, self.k_vector
                        )
                        overlap_integral = self.get_energy_integral(
                            neighbour, orbital, ket_atom, ket_orbital
                        )
                        self.energy_integral_symbols.update(
                            overlap_integral.free_symbols
                        )
                        matrix[i, j] += Mul(bloch_factor, overlap_integral)

        return matrix

    def _get_soc_matrix(self) -> Matrix:
        block_diagnonal_soc_matrices = []
        for atom in self.unit_cell.atoms:
            for orbital_class, orbitals in atom.active_orbitals.items():
                # Create spin orbit matrix
                spins: Tuple[Spin, ...] = tuple(Spin)
                _orbitals: List[Orbital] = orbitals
                orbital_basis: Tuple[Tuple[Orbital, Spin], ...] = tuple(
                    itertools.product(_orbitals, spins)
                )
                if orbital_class in atom.spin_coupled_orbitals:
                    soc_matrix = get_soc_matrix(orbital_basis, atom)
                    self.soc_symbols.update(soc_matrix.free_symbols)
                    block_diagnonal_soc_matrices.append(soc_matrix)
                else:
                    block_diagnonal_soc_matrices.extend([0] * len(orbital_basis))

        return diag(*block_diagnonal_soc_matrices)

    def get_energy_symbol(self, atom: Atom, orbital: Orbital) -> Symbol:
        return Symbol(f"E_{atom.type + orbital.orbital_class.name}", real=True)

    def get_energy_integral(
        self,
        bra_atom: Neighbour,
        bra_orbital: Orbital,
        ket_atom: Atom,
        ket_orbital: Orbital,
    ) -> Expr:
        collection = ParamCollection(
            bra_atom.atom.type, ket_atom.type, self.coord_symbols, self.angle_symbols
        )
        energy_integral = collection.get_energy_integral(
            bra_atom.atom, bra_orbital, ket_atom, ket_orbital
        )

        direction_cosine_vec = self.direction_cosine(
            ket_atom.coordinate, bra_atom.observed_coordinate
        )
        direction_cosine_tuple = sympy_unpack(direction_cosine_vec, self.r)

        for i in range(3):
            energy_integral = energy_integral.subs(
                self.angle_symbols[i], direction_cosine_tuple[i]
            )

        return energy_integral

    @staticmethod
    def direction_cosine(coord1: BaseVector, coord2: BaseVector) -> Expr:
        coord1 = coord1
        coord2 = coord2
        rel_vec = coord2 - coord1
        distance = sqrt(rel_vec.dot(rel_vec))

        output = rel_vec / distance
        return output

    def get_overlap_parameter_symbols(self) -> Set[Symbol]:
        overlap_symbols: Set[Symbol] = set()
        for x in self.overlap_integrals.values():
            overlap_symbols = overlap_symbols.union(
                set(x.two_center_integrals.values())
            )

        return overlap_symbols

    def get_energy_parameter_symbols(self) -> Set[Symbol]:
        return set(self.energies.values())

    def get_parameter_symbols(self) -> Set[Symbol]:
        return (
            self.get_energy_parameter_symbols()
            | self.get_overlap_parameter_symbols()
            | self.soc_symbols
        )

    def get_hamiltonians(
        self, parameters: Dict[Symbol, Number], k_values: npt.NDArray[np.float_]
    ) -> npt.NDArray[np.float_]:
        matrix = self.construct_hamiltonian(1)
        substitutions = {}
        for symbol in matrix.free_symbols.difference(self.k_symbols):
            substitutions[symbol] = parameters[str(symbol)]

        # Hacky optimization to replace constant zeros with a variable so we
        # can do matrix operations on the entire collection of matrices instead
        # of one by one.
        z = symbols("z")
        for i, j in itertools.product(range(matrix.shape[0]), repeat=2):
            if matrix[i, j] == 0:
                matrix[i, j] = z

        variables = list(itertools.chain(substitutions.keys(), ["kx", "ky", "kz"], [z]))
        lambda_matrix: Callable[..., npt.NDArray[np.float_]] = lambdify(variables, matrix)

        params = np.tile(
            np.atleast_2d(list(substitutions.values())).T, (1, k_values.shape[1])
        )
        zeros = np.zeros(k_values.shape[1])

        return lambda_matrix(*params, *k_values, zeros)

    def get_energy_eigenvalues(
        self, parameters: Dict[Symbol, Number], k_values: npt.NDArray[np.float_]
    ) -> npt.NDArray[np.float_]:
        """Retreive energy eigenvalues for a give array of wavevectors."""
        matrices = self.get_hamiltonians(parameters, k_values)
        return LA.eigvalsh(matrices.T).T

    def get_energy_eigen(
        self, parameters: Dict[Symbol, Number], k_values: npt.NDArray[np.float_]
    ) -> Tuple[npt.NDArray[np.float_], npt.NDArray[np.float_]]:
        """Retreive both energy eigenfunc and eigenval for an array of wavevectors."""
        matrices = self.get_hamiltonians(parameters, k_values)
        transposed_eigenvalues, transposed_eigenfunctions = LA.eigh(matrices.T)
        return transposed_eigenvalues.T, transposed_eigenfunctions.T


def sympy_unpack(
    vector: BaseVector, coordsystem: CoordSys3D
) -> Tuple[Number, Number, Number]:
    return (
        vector.dot(coordsystem.i),
        vector.dot(coordsystem.j),
        vector.dot(coordsystem.k),
    )


def calculate_bloch_factor(r: CoordSys3D, k: BaseVector) -> Expr:
    return exp(I * dot(k, r))


def sequence_to_sympy_vector(r: CoordSys3D, array: npt.NDArray[np.float_]) -> Expr:
    assert len(array) == 3
    return array[0] * r.i + array[1] * r.j + array[2] * r.k
