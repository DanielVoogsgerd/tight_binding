"""Collection of objects for a Tight Binding model."""
from dataclasses import dataclass
import numpy as np
import numpy.typing as npt
from sympy import sqrt
from sympy.vector import CoordSys3D, BaseVector
from itertools import product, chain
from fractions import Fraction
import json
from collections import defaultdict
from enum import Enum

from typing import Tuple, List, Dict, Iterable, Union, Sequence, Self, Optional, Any

Number = Union[float, int]
Coord = Tuple[float, float, float]
NPCoord = npt.NDArray[np.float_]

r = CoordSys3D("r")

ORBITAL_MAP = {
    "s": ["s"],
    "p": ["px", "py", "pz"],
    "d": ["xz", "yz", "xy", "x2-y2", "3z2-r2"],
}


class Spin(Enum):
    """Quantum mechanical Spin half enumeration."""

    DOWN = -Fraction(1 / 2)
    UP = Fraction(1 / 2)


class OrbitalClass(Enum):
    """Orbital Class enumeration."""

    s = 0
    p = 1
    d = 2
    f = 3

    @classmethod
    def has_letter(cls, letter: str) -> bool:
        """Check if enumeration has a certain name."""
        return letter in cls.__members__


@dataclass(frozen=True, eq=True)
class Orbital:
    """Quantum mechanical orbital."""

    orbital_map = {
        OrbitalClass.s: ["s"],
        OrbitalClass.p: ["px", "py", "pz"],
        OrbitalClass.d: ["xz", "yz", "xy", "x2-y2", "3z2-r2"],
    }

    representation: str

    @classmethod
    def from_orbital_class(cls, orbital_class: OrbitalClass) -> List[Self]:
        """Generate List of Orbitals from a certain azimuthal quantum number."""
        orbitals = []
        for orbital in cls.orbital_map[orbital_class]:
            orbitals.append(cls(orbital))

        return orbitals

    @property
    def parity(self) -> int:
        """Partity of the Orbital."""
        return self.azimuthal_quantum_number % 2

    @property
    def azimuthal_quantum_number(self) -> int:
        """Azimuthal quantum number of the Orbital."""
        for i, (orbital_class, orbitals) in enumerate(self.orbital_map.items()):
            if self.representation in orbitals:
                return i

        raise Exception("Orbital has no known quantum number")

    @property
    def l(self) -> int: # noqa: E743
        """Azimuthal quantum number of the Orbital."""
        return self.azimuthal_quantum_number

    @property
    def orbital_class(self) -> OrbitalClass:
        """Get historical letter representing the azimuthal quantum number."""
        return OrbitalClass(self.azimuthal_quantum_number)

    def __str__(self) -> str:
        """Return the geometric representation of the orbital."""
        return self.representation


@dataclass(frozen=True)
class Atom:
    """Object representation of an Atom."""

    type: str
    coordinate: BaseVector
    valence_orbitals: Tuple[Tuple[OrbitalClass, Tuple[Orbital, ...]], ...]
    conductance_orbitals: Tuple[Tuple[OrbitalClass, Tuple[Orbital, ...]], ...]
    spin_coupled_orbitals: Tuple[OrbitalClass, ...]
    flipped_odd_orbitals: bool = False

    @property
    def active_orbitals(self) -> Dict[OrbitalClass, List[Orbital]]:
        """Orbitals that are enabled for a certain Atom."""
        all_orbitals: Dict[OrbitalClass, List[Orbital]] = defaultdict(list)

        for orbital_class, orbitals in self.valence_orbitals:
            all_orbitals[orbital_class].extend(orbitals)
        for orbital_class, orbitals in self.conductance_orbitals:
            all_orbitals[orbital_class].extend(orbitals)

        return all_orbitals


# TODO: Replace with AtomObservation that inherits Atom
@dataclass
class Neighbour:
    """An observation of an Atom at a certain location."""

    atom: Atom
    observed_coordinate: NPCoord


class UnitCell:
    """Unit Cell object."""

    lattice_vectors: NPCoord
    atom_map: Dict[int, Atom]
    atoms: List[Atom]
    id_map: Dict[Atom, int]
    spin_orbit_coupling: bool = False
    conductance_orbital_count: int = 0
    valence_orbital_count: int = 0

    def __init__(
        self,
        lattice_vectors: Sequence[BaseVector],
        brillouin_zone: Optional[Dict[str, Coord]],
    ) -> None:
        """Create a Unit Cell object."""
        self.lattice_vectors = np.array(lattice_vectors)
        self.int = 0
        self.atoms = []
        self.brillouin_zone = brillouin_zone

    @classmethod
    def from_dict(cls, structure: Dict[str, Any], r: CoordSys3D) -> Self:
        """Initialise Unit Cell from dict."""
        lattice_vectors = []
        for lattice_vector in structure["lattice"]:
            lattice_vectors.append(
                Fraction(lattice_vector[0]) * r.i
                + Fraction(lattice_vector[1]) * r.j
                + Fraction(lattice_vector[2]) * r.k
            )

        brillouin_zone = structure["brillouin_zone"]

        unit_cell = cls(lattice_vectors, brillouin_zone)

        ORBITALS = list(chain.from_iterable(ORBITAL_MAP.values()))

        for atom in structure["basis"]:
            for position in atom["positions"]:
                sorted_orbitals: Dict[str, Dict[OrbitalClass, List[Orbital]]] = {
                    "valence": {
                        orbital_class: list() for orbital_class in OrbitalClass
                    },
                    "conductance": {
                        orbital_class: list() for orbital_class in OrbitalClass
                    },
                }
                for active_orbital_type in atom["conductance_orbitals"]:
                    if OrbitalClass.has_letter(active_orbital_type):
                        orbital_class = OrbitalClass[active_orbital_type]
                        orbitals = Orbital.from_orbital_class(orbital_class)
                        unit_cell.conductance_orbital_count += len(orbitals)
                        sorted_orbitals["conductance"][orbital_class].extend(orbitals)
                    elif active_orbital_type in ORBITALS:
                        orbital = Orbital(active_orbital_type)
                        sorted_orbitals["conductance"][orbital.orbital_class].append(
                            orbital
                        )
                        unit_cell.conductance_orbital_count += 1
                    else:
                        raise ValueError(
                            "Active orbital: {:s} does not exist".format(
                                active_orbital_type
                            )
                        )

                for active_orbital_type in atom["valence_orbitals"]:
                    if OrbitalClass.has_letter(active_orbital_type):
                        orbital_class = OrbitalClass[active_orbital_type]
                        orbitals = Orbital.from_orbital_class(orbital_class)
                        unit_cell.valence_orbital_count += len(orbitals)
                        sorted_orbitals["valence"][orbital_class].extend(orbitals)
                    elif active_orbital_type in ORBITALS:
                        orbital = Orbital(active_orbital_type)
                        sorted_orbitals["valence"][orbital.orbital_class].append(
                            orbital
                        )
                        unit_cell.valence_orbital_count += 1
                    else:
                        raise ValueError(
                            "Active orbital: {:s} does not exist".format(
                                active_orbital_type
                            )
                        )

                flip_odd = (
                    True
                    if "flip_odd_orbitals" in atom and atom["flip_odd_orbitals"]
                    else False
                )

                spin_coupled_orbitals = (
                    tuple(
                        (
                            OrbitalClass[orbital_class_letter]
                            for orbital_class_letter in atom["spin_coupled_orbitals"]
                        )
                    )
                    if "spin_coupled_orbitals" in atom
                    else ()
                )

                if len(spin_coupled_orbitals) > 0:
                    unit_cell.spin_orbit_coupling = True

                atom_instance = Atom(
                    atom["type"],
                    tuple_to_vector(position, r),
                    tuple((k, tuple(v)) for k, v in sorted_orbitals["valence"].items()),
                    tuple(
                        (k, tuple(v)) for k, v in sorted_orbitals["conductance"].items()
                    ),
                    spin_coupled_orbitals,
                    flip_odd,
                )
                unit_cell.add_atom(atom_instance)

        return unit_cell

    @classmethod
    def from_file(cls, filename: str, r: CoordSys3D) -> Self:
        """Load unit cell from file."""
        with open(filename, encoding="utf-8") as f:
            return cls.from_dict(json.loads(f.read()), r)

    def add_atom(self, atom: Atom) -> None:
        """Add atom in the unit cell."""
        self.atoms.append(atom)

    def add_atoms(self, atoms: Iterable[Atom]) -> None:
        """Add multiple atoms to the unit cell."""
        for atom in atoms:
            self.add_atom(atom)

    def get_symmetry_points(self, symmetry_point_letters: Iterable[str]) -> npt.NDArray[np.float_]:
        """Get list of symmetry points."""
        if not self.brillouin_zone:
            raise RuntimeError(
                "Cannot use get_symmetry_points()"
                "on Unit Cells with no defined Brillouin Zone."
            )

        return np.array(
            [
                np.array(self.brillouin_zone[letter]) * 2 * np.pi
                for letter in symmetry_point_letters
            ]
        )

    def get_k_values(
        self,
        symmetry_points: Sequence[NPCoord],
        points_per_directions: Sequence[int],
    ) -> npt.NDArray[np.float_]:
        """Return k_vectors between a list of symmetry points."""
        transitions = tuple(zip(symmetry_points[:-1], symmetry_points[1:]))
        assert len(transitions) == len(points_per_directions)

        k_values_list = []
        for transition, points_per_direction in zip(transitions, points_per_directions):
            k_values = np.linspace(*transition, points_per_direction, axis=1)
            k_values_list.append(k_values)

        k_values = np.concatenate(k_values_list, axis=1)

        return k_values

    # TODO: Seems duplicate of method above
    def get_parameter_variable(
        self, points_per_directions: Sequence[int]
    ) -> npt.NDArray[np.float_]:
        """Return k_vectors between a list of symmetry points."""
        k_values_list = []
        for i, points_per_direction in enumerate(points_per_directions):
            k_values = np.linspace(i, i + 1, points_per_direction)
            k_values_list.append(k_values)

        k_values = np.concatenate(k_values_list)

        return k_values

    def sort_neighbours(self) -> List[Tuple[float, Dict[Atom, List[Neighbour]]]]:
        """Create a list of neighbouring pairs sorted and and collected by distance."""
        neighbour_pairs: Dict[float, Dict[Atom, List[Neighbour]]] = defaultdict(
            lambda: defaultdict(list)
        )

        atoms = self.atoms
        for i, atom in enumerate(atoms):
            for lattice_vector in product([-1, 0, 1], repeat=3):
                for possible_neighbour in atoms:
                    # TODO: use += and don't use ii
                    new_coord = possible_neighbour.coordinate
                    for ii in range(3):
                        new_coord = (
                            new_coord + self.lattice_vectors[ii] * lattice_vector[ii]
                        )

                    relative_coordinate = new_coord - atom.coordinate

                    distance = sqrt(relative_coordinate.dot(relative_coordinate))

                    if distance == 0:
                        continue

                    neighbour_pairs[distance][atom].append(
                        Neighbour(possible_neighbour, new_coord)
                    )

        sorted_neighbourpairs = []
        distances: Iterable[float] = sorted(neighbour_pairs.keys())
        for distance in distances:
            sorted_neighbourpairs.append((distance, neighbour_pairs[distance]))

        return sorted_neighbourpairs


def tuple_to_vector(tuple: Sequence[Number], r: CoordSys3D) -> BaseVector:
    """Convert sequence of length 3 to 3D vector."""
    return (
        Fraction(tuple[0]) * r.i + Fraction(tuple[1]) * r.j + Fraction(tuple[2]) * r.k
    )
