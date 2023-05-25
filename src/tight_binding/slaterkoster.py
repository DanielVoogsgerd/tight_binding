from sympy import symbols, sqrt, Symbol, Expr
from typing import Tuple, Dict, List

from tight_binding.objects import Orbital, Atom


class ParamCollection:
    """A collection of all [s-d] - [s-d] interactions in sympy"""

    coords = ["x", "y", "z"]

    s_orbitals = ["s"]
    p_orbitals = ["px", "py", "pz"]
    d_orbitals = ["xz", "yz", "xy", "x2-y2", "3z2-r2"]

    orbital_type = {
        0: "s",
        1: "p",
        2: "d",
    }

    def __init__(
        self,
        atom1: str,
        atom2: str,
        coord_symbols: Tuple[Symbol, Symbol, Symbol],
        angle_symbols: Tuple[Symbol, Symbol, Symbol],
    ) -> None:
        self.coord_symbols = coord_symbols
        self.angle_symbols = angle_symbols

        self.two_center_integrals: Dict[Tuple[str, str, str, str, str], Symbol] = {}
        self.all_orbitals = self.s_orbitals + self.p_orbitals + self.d_orbitals

        self.generate_two_center_integrals(atom1, atom2)

        l, m, n = self.angle_symbols # noqa: E741

        self.table = {
            ("s", "s"): (1, 0, 0),
            ("s", "px"): (l, 0, 0),
            ("s", "py"): (m, 0, 0),
            ("s", "pz"): (n, 0, 0),
            ("px", "px"): (l**2, 1 - l**2, 0),
            ("px", "py"): (l * m, -l * m, 0),
            ("px", "pz"): (l * n, -l * n, 0),
            ("py", "px"): (l * m, -l * m, 0),
            ("py", "py"): (m**2, 1 - m**2, 0),
            ("py", "pz"): (n * m, -n * m, 0),
            ("pz", "px"): (l * n, l * n, 0),
            ("pz", "py"): (n * m, -n * m, 0),
            ("pz", "pz"): (n**2, 1 - n**2, 0),
            ("s", "xy"): (sqrt(3) * l * m, 0, 0),
            ("s", "yz"): (sqrt(3) * m * n, 0, 0),
            ("s", "xz"): (sqrt(3) * l * n, 0, 0),
            ("s", "x2-y2"): (1 / 2 * sqrt(3) * (l**2 - m**2), 0, 0),
            ("s", "3z2-r2"): (n**2 - 1 / 2 * (l**2 + m**2), 0, 0),
            ("px", "xy"): (sqrt(3) * l**2 * m, m * (1 - 2 * l**2), 0),
            ("px", "yz"): (sqrt(3) * l * m * n, -2 * l * m * n, 0),
            ("px", "xz"): (sqrt(3) * l**2 * n, n * (1 - 2 * l**2), 0),
            ("py", "xy"): (sqrt(3) * l * m**2, l * (1 - 2 * m**2), 0),
            ("py", "yz"): (sqrt(3) * m**2 * n, n * (1 - 2 * m**2), 0),
            ("py", "xz"): (sqrt(3) * l * m * n, -2 * l * m * n, 0),
            ("pz", "xy"): (sqrt(3) * l * m * n, -2 * l * m * n, 0),
            ("pz", "yz"): (sqrt(3) * n**2 * m, m * (1 - 2 * n**2), 0),
            ("pz", "xz"): (sqrt(3) * n**2 * l, l * (1 - 2 * n**2), 0),
            ("px", "x2-y2"): (
                1 / 2 * sqrt(3) * l * (l**2 - m**2),
                l * (1 - l**2 + m**2),
                0,
            ),
            ("py", "x2-y2"): (
                1 / 2 * sqrt(3) * m * (l**2 - m**2),
                -m * (1 + l**2 - m**2),
                0,
            ),
            ("pz", "x2-y2"): (
                1 / 2 * sqrt(3) * n * (l**2 - m**2),
                -n * (l**2 - m**2),
                0,
            ),
            ("px", "3z2-r2"): (
                l * (n**2 - 1 / 2 * (l**2 + m**2)),
                -sqrt(3) * l * n**2,
                0,
            ),
            ("py", "3z2-r2"): (
                m * (n**2 - 1 / 2 * (l**2 + m**2)),
                -sqrt(3) * m * n**2,
                0,
            ),
            ("pz", "3z2-r2"): (
                n * (n**2 - 1 / 2 * (l**2 + m**2)),
                +sqrt(3) * n * (l**2 + m**2),
                0,
            ),
            ("xy", "xy"): (
                (3 * l**2 * m**2),
                (l**2 + m**2 - 4 * l**2 * m**2),
                (n**2 + l**2 * m**2),
                0,
            ),
            ("xy", "yz"): (
                (3 * l * m**2 * n),
                (l * n * (1 - 4 * m**2)),
                (l * n * (m**2 - 1)),
                0,
            ),
            ("xy", "xz"): (
                (3 * l**2 * m * n),
                (m * n * (1 - 4 * l**2)),
                (m * n * (l**2 - 1)),
                0,
            ),
            ("xy", "x2-y2"): (
                (3 / 2 * l * m * (l**2 - m**2)),
                (2 * l * m * (m**2 - l**2)),
                (1 / 2 * l * m * (l**2 - m**2)),
            ),
            ("xy", "3z2-r2"): (
                (sqrt(3) * l * m * (n**2 - 1 / 2 * (l**2 + m**2))),
                (-2 * sqrt(3) * l * m * n**2),
                (1 / 2 * sqrt(3) * l * m * (1 + n**2)),
            ),
            ("yz", "xy"): (
                (3 * l * m**2 * n),
                (l * n * (1 - 4 * m**2)),
                (l * n * (m**2 - 1)),
            ),
            ("yz", "yz"): (
                (3 * m**2 * n**2),
                (m**2 + n**2 - 4 * m**2 * n**2),
                (l**2 + m**2 * n**2),
            ),
            ("yz", "xz"): (
                (3 * m * n**2 * l),
                (m * l * (1 - 4 * n**2)),
                (m * l * (n**2 - 1)),
            ),
            ("yz", "x2-y2"): (
                (3 / 2 * m * n * (l**2 - m**2)),
                (-m * n * (1 + 2 * (l**2 - m**2))),
                (m * n * (1 + 1 / 2 * (l**2 - m**2))),
            ),
            ("yz", "3z2-r2"): (
                (sqrt(3) * m * n * (n**2 - 1 / 2 * (l**2 + m**2))),
                (sqrt(3) * m * n * (l**2 + m**2 - n**2)),
                (-1 / 2 * sqrt(3) * m * n * (l**2 + m**2)),
            ),
            ("xz", "xy"): (
                (3 * l**2 * m * n),
                (m * n * (1 - 4 * l**2)),
                (m * n * (l**2 - 1)),
            ),
            ("xz", "yz"): (
                (3 * m * n**2 * l),
                (m * l * (1 - 4 * n**2)),
                (m * l * (n**2 - 1)),
            ),
            ("xz", "xz"): (
                (3 * l**2 * n**2),
                (l**2 + n**2 - 4 * l**2 * n**2),
                (m**2 + l**2 * n**2),
            ),
            ("xz", "x2-y2"): (
                (3 / 2 * n * l * (l**2 - m**2)),
                (n * l * (1 - 2 * (l**2 - m**2))),
                (-n * l * (1 - 1 / 2 * (l**2 - m**2))),
            ),
            ("xz", "3z2-r2"): (
                (sqrt(3) * l * n * (n**2 - 1 / 2 * (l**2 + m**2))),
                (sqrt(3) * l * n * (l**2 + m**2 - n**2)),
                (-1 / 2 * sqrt(3) * l * n * (l**2 + m**2)),
            ),
            ("x2-y2", "xy"): (
                (3 / 2 * l * m * (l**2 - m**2)),
                (2 * l * m * (m**2 - l**2)),
                (1 / 2 * l * m * (l**2 - m**2)),
            ),
            ("x2-y2", "yz"): (
                (3 / 2 * m * n * (l**2 - m**2)),
                (-m * n * (1 + 2 * (l**2 - m**2))),
                (m * n * (1 + 1 / 2 * (l**2 - m**2))),
            ),
            ("x2-y2", "xz"): (
                (3 / 2 * n * l * (l**2 - m**2)),
                (n * l * (1 - 2 * (l**2 - m**2))),
                (-n * l * (1 - 1 / 2 * (l**2 - m**2))),
            ),
            ("x2-y2", "x2-y2"): (
                (3 / 4 * (l**2 - m**2) ** 2),
                (l**2 + m**2 - (l**2 - m**2) ** 2),
                (n**2 + 1 / 4 * (l**2 - m**2) ** 2),
            ),
            ("x2-y2", "3z2-r2"): (
                (
                    (1 / 2 * sqrt(3))
                    * (l**2 - m**2)
                    * (n**2 - 1 / 2 * (l**2 + m**2))
                ),
                (sqrt(3) * n**2 * (m**2 - l**2)),
                (1 / 4 * sqrt(3) * (1 + n**2) * (l**2 - m**2)),
            ),
            ("3z2-r2", "xy"): (
                (sqrt(3) * l * m * (n**2 - 1 / 2 * (l**2 + m**2))),
                (-2 * sqrt(3) * l * m * n**2),
                (1 / 2 * sqrt(3) * l * m * (1 + n**2)),
            ),
            ("3z2-r2", "yz"): (
                (sqrt(3) * m * n * (n**2 - 1 / 2 * (l**2 + m**2))),
                (sqrt(3) * m * n * (l**2 + m**2 - n**2)),
                (-1 / 2 * sqrt(3) * m * n * (l**2 + m**2)),
            ),
            ("3z2-r2", "xz"): (
                (sqrt(3) * l * n * (n**2 - 1 / 2 * (l**2 + m**2))),
                (sqrt(3) * l * n * (l**2 + m**2 - n**2)),
                (-1 / 2 * sqrt(3) * l * n * (l**2 + m**2)),
            ),
            ("3z2-r2", "x2-y2"): (
                (
                    (1 / 2 * sqrt(3))
                    * (l**2 - m**2)
                    * (n**2 - 1 / 2 * (l**2 + m**2))
                ),
                (sqrt(3) * n**2 * (m**2 - l**2)),
                (1 / 4 * sqrt(3) * (1 + n**2) * (l**2 - m**2)),
            ),
            ("3z2-r2", "3z2-r2"): (
                (n**2 - 1 / 2 * (l**2 + m**2)) ** 2,
                (3 * n**2 * (l**2 + m**2)),
                (3 / 4 * (l**2 + m**2) ** 2),
            ),
        }

    def generate_two_center_integrals(self, atom1_type: str, atom2_type: str) -> None:
        overlap_integral_type = {
            "ssσ",
            "spσ",
            "ppσ",
            "ppπ",
            "sdσ",
            "pdσ",
            "pdπ",
            "ddσ",
            "ddπ",
            "ddδ",
        }

        for overlap_int_type in overlap_integral_type:
            atom1, atom2 = sorted([atom1_type, atom2_type])

            identifier = (
                atom1
                + overlap_int_type[0]
                + atom2
                + overlap_int_type[1]
                + overlap_int_type[2]
            )
            key = (
                atom1,
                overlap_int_type[0],
                atom2,
                overlap_int_type[1],
                overlap_int_type[2],
            )
            self.two_center_integrals[key] = symbols(identifier, real=True)

            if overlap_int_type[0] != overlap_int_type[1]:
                identifier = (
                    atom2
                    + overlap_int_type[0]
                    + atom1
                    + overlap_int_type[1]
                    + overlap_int_type[2]
                )
                key = (
                    atom2,
                    overlap_int_type[0],
                    atom1,
                    overlap_int_type[1],
                    overlap_int_type[2],
                )
                self.two_center_integrals[key] = symbols(identifier, real=True)

    def get_two_center_integrals(
        self, bra_atom: Atom, bra_orbital: Orbital, ket_atom: Atom, ket_orbital: Orbital
    ) -> List[Symbol]:
        directions = ["σ", "π", "δ"]

        bra_atom_type = bra_atom.type
        ket_atom_type = ket_atom.type

        if bra_orbital.l == ket_orbital.l:
            orbital_class = bra_orbital.orbital_class
            atom1, atom2 = sorted([bra_atom_type, ket_atom_type])

            identifier = atom1 + orbital_class.name + atom2 + orbital_class.name
        elif bra_orbital.l < ket_orbital.l:
            identifier = (
                bra_atom.type
                + bra_orbital.orbital_class.name
                + ket_atom_type
                + ket_orbital.orbital_class.name
            )
        else:
            identifier = (
                ket_atom_type
                + ket_orbital.orbital_class.name
                + bra_atom_type
                + bra_orbital.orbital_class.name
            )

        return [Symbol(identifier + direction, real=True) for direction in directions]

    def get_energy_integral(
        self, bra_atom: Atom, bra_orbital: Orbital, ket_atom: Atom, ket_orbital: Orbital
    ) -> Expr:
        flipped = (
            bra_orbital.azimuthal_quantum_number > ket_orbital.azimuthal_quantum_number
        )

        if flipped:
            factors = self.table[(str(ket_orbital), str(bra_orbital))]
        else:
            factors = self.table[(str(bra_orbital), str(ket_orbital))]

        integrals = self.get_two_center_integrals(
            bra_atom, bra_orbital, ket_atom, ket_orbital
        )

        parity = (
            -1
            if bra_orbital.l > ket_orbital.l
            and (bra_orbital.l + ket_orbital.l) % 2 == 1
            else 1
        )

        flip = -1 if bra_orbital.l % 2 == 1 and bra_atom.flipped_odd_orbitals else 1
        flip *= -1 if ket_orbital.l % 2 == 1 and ket_atom.flipped_odd_orbitals else 1

        term1 = parity * flip * (factors[0] * integrals[0])
        term2 = parity * flip * (factors[1] * integrals[1])
        term3 = parity * flip * (factors[2] * integrals[2])

        return term1 + term2 + term3
