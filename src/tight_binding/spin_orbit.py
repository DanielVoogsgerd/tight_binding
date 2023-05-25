from sympy import Symbol, sqrt, Expr, I, Matrix
from itertools import product

import sympy.matrices as matrices

from collections import OrderedDict
from dataclasses import dataclass, replace
from fractions import Fraction

from tight_binding.objects import Spin, Atom, Orbital, OrbitalClass

from typing import List, Tuple, Sequence, Dict, Union, Self, Any

Number = Union[int, float]
Harmonic_Map = Dict[str, List[Tuple[Union[Expr, Number], Tuple[int, int]]]]


@dataclass
class State:
    """A combined angular momentum/spin state."""

    l: int # noqa: E741
    ml: int
    s: Fraction
    ms: Fraction

    def Lp(self) -> Expr:
        """Apply the angular momentum plus operator and return the factor."""
        factor = sqrt(self.l * (self.l + 1) - self.ml * (self.ml + 1))
        self.ml += 1
        return factor

    def Lm(self) -> Expr:
        """Apply the angular momentum minus operator and return the factor."""
        factor = sqrt(self.l * (self.l + 1) - self.ml * (self.ml - 1))
        self.ml -= 1
        return factor

    def Lz(self) -> Expr:
        """Apply the angular momentum z operator and return the factor."""
        return self.ml

    def Sp(self) -> Expr:
        """Apply the spin plus operator and return the factor."""
        factor = sqrt(self.s * (self.s + 1) - self.ms * (self.ms + 1))
        self.ms += 1
        return factor

    def Sm(self) -> Expr:
        """Apply the spin minus operator and return the factor."""
        factor = sqrt(self.s * (self.s + 1) - self.ms * (self.ms - 1))
        self.ms -= 1
        return factor

    def Sz(self) -> Expr:
        """Apply the spin z operator and return the factor."""
        return self.ms

    def clone(self) -> Expr:
        """Shallow clone the State."""
        return replace(self)

    def __eq__(s: Self, o: Any) -> bool:
        """Compare if the state is equal."""
        if not isinstance(o, State):
            raise NotImplementedError(
                f'Comparison between "{type(s)}" and "{type(o)}" is not supported'
            )

        return s.l == o.l and s.ml == o.ml and s.s == o.s and s.ms == o.ms

    def is_legal(self) -> bool:
        """Check if a state abides by the laws of physics."""
        return abs(self.ml) <= self.l and abs(self.ms) <= self.s


states_content = list(
    product(
        [1, 2],
        [-2, -1, 0, 1, 2],
        [Fraction(1 / 2)],
        [Fraction(1 / 2), Fraction(-1 / 2)],
    )
)

states = [State(*state) for state in states_content if abs(state[1]) <= state[0]]

s_up = (Fraction(1 / 2), Fraction(1 / 2))
s_down = (Fraction(1 / 2), -Fraction(1 / 2))

s_orbital_harmonic_map: Harmonic_Map = OrderedDict([("s", [(1, (0, 0))])])

p_orbital_harmonic_map: Harmonic_Map = OrderedDict(
    [
        ("px", [(-1 / sqrt(2), (1, 1)), (1 / sqrt(2), (1, -1))]),
        ("py", [(I / sqrt(2), (1, 1)), (I / sqrt(2), (1, -1))]),
        (
            "pz",
            [
                (1, (1, 0)),
            ],
        ),
    ]
)

d_orbital_harmonic_map: Harmonic_Map = OrderedDict(
    [
        ("xz", [(-1 / sqrt(2), (2, 1)), (1 / sqrt(2), (2, -1))]),
        ("yz", [(I / sqrt(2), (2, 1)), (I / sqrt(2), (2, -1))]),
        ("x2-y2", [(1 / sqrt(2), (2, 2)), (1 / sqrt(2), (2, -2))]),
        ("xy", [(-I / sqrt(2), (2, 2)), (I / sqrt(2), (2, -2))]),
        (
            "3z2-r2",
            [
                (1, (2, 0)),
            ],
        ),
    ]
)


orbital_harmonic_map = OrderedDict()
orbital_harmonic_map.update(s_orbital_harmonic_map)
orbital_harmonic_map.update(p_orbital_harmonic_map)
orbital_harmonic_map.update(d_orbital_harmonic_map)

test_basis = [
    ("px", Spin.UP),
    ("py", Spin.UP),
    ("pz", Spin.UP),
    ("px", Spin.DOWN),
    ("py", Spin.DOWN),
    ("pz", Spin.DOWN),
]

ORBITAL_MAP = {
    "s": ["s"],
    "p": ["px", "py", "pz"],
    "d": ["xz", "yz", "xy", "x2-y2", "3z2-r2"],
}


def get_soc_basis(
    orbital_basis: Sequence[Tuple[Orbital, Spin]], atom: Atom
) -> List[State]:
    # Check if the basis is unique
    assert len(orbital_basis) == len(set(orbital_basis))

    l_values = [orbital_class.value for orbital_class in atom.spin_coupled_orbitals]

    states_content = list(
        product(
            l_values,
            [-2, -1, 0, 1, 2],
            [Fraction(1 / 2)],
            [Fraction(1 / 2), Fraction(-1 / 2)],
        )
    )

    states = [State(*state) for state in states_content if abs(state[1]) <= state[0]]

    return states


def get_harmonic_soc_matrix(soc_basis: List[State], atom: Atom) -> Matrix:
    lpsm_matrix = matrices.zeros(len(soc_basis))
    lmsp_matrix = matrices.zeros(len(soc_basis))
    lzsz_matrix = matrices.zeros(len(soc_basis))

    lambdas = {
        orbital_class.value: Symbol(
            "lambda_{atom},{orbital_class}".format(
                atom=atom.type, orbital_class=orbital_class.name
            ),
            real=True,
        )
        for orbital_class in OrbitalClass
    }

    for n, state in enumerate(soc_basis):
        factor = 1
        st = state.clone()
        factor *= st.Lp()
        factor *= st.Sm()

        if st.is_legal():
            m = soc_basis.index(st)
            lpsm_matrix[n, m] = factor * lambdas[st.l] / 2

    for n, state in enumerate(soc_basis):
        factor = 1
        st = state.clone()
        factor *= st.Lm()
        factor *= st.Sp()

        if st.is_legal():
            m = soc_basis.index(st)
            lmsp_matrix[n, m] = factor * lambdas[st.l] / 2

    for n, state in enumerate(soc_basis):
        factor = 1
        st = state.clone()
        factor *= st.Lz()
        factor *= st.Sz()

        if st.is_legal():
            m = soc_basis.index(st)
            lzsz_matrix[n, m] = factor * lambdas[st.l] / 2

    H_soc = lpsm_matrix + lmsp_matrix + 2 * lzsz_matrix

    return H_soc


def get_soc_matrix(orbital_basis: Sequence[Tuple[Orbital, Spin]], atom: Atom) -> Matrix:
    for orbital_class in atom.spin_coupled_orbitals:
        needed_basis_elements = set(
            product(Orbital.from_orbital_class(orbital_class), Spin)
        )
        assert needed_basis_elements.issubset(orbital_basis)

    soc_basis = get_soc_basis(orbital_basis, atom)
    H_soc = get_harmonic_soc_matrix(soc_basis, atom)

    T = matrices.zeros(len(orbital_basis), len(soc_basis))
    for n, (orbital, spin) in enumerate(orbital_basis):
        for factor, harmonic_content in orbital_harmonic_map[str(orbital)]:
            harmonic = State(*harmonic_content, Fraction(1 / 2), spin.value)
            m = soc_basis.index(harmonic)
            T[m, n] = factor

    return T.inv() * H_soc * T
