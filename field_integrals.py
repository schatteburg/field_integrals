import numpy as np
from matplotlib import pyplot as plt
import os
from scipy.integrate import quad
from scipy.interpolate import RegularGridInterpolator, CloughTocher2DInterpolator
import math
import pickle
import pandas as pd

from collections.abc import Sequence
from typing import Union

######### decorators #########
def check_field_compatibility_strict(func: callable) -> callable:
    def wrapper(self, other: "field", *args, **kwargs):
        if not self.coordinates == other.coordinates:
            raise ValueError(f"Coordinates of fields don't match!")
        elif not self.shape == other.shape:
            raise ValueError(f"Shapes of fields don't match: {self.shape} and {other.shape}.")
        else:
            return func(self, other, *args, **kwargs)

    return wrapper

def check_field_compatibility_incl_numbers(func: callable) -> callable:
    def wrapper(self, other: Union["field",int,float,complex], *args, **kwargs):
        if isinstance(other, (int,float,complex)):
            return func(self, other, *args, **kwargs)
        if not self.coordinates == other.coordinates:
            raise ValueError(f"Coordinates of fields don't match!")
        elif not self.shape == other.shape:
            raise ValueError(f"Shapes of fields don't match: {self.shape} and {other.shape}.")
        else:
            return func(self, other, *args, **kwargs)

    return wrapper

def todo_error():
    raise NotImplementedError(f"Please implement this method (completely).")


class field():

    def __init__(self, fieldvalues: np.ndarray, coordinates: dict[str, Sequence[float]], units: dict[str, str] = None, coordinate_system: str = None, name: str = None, vocal: bool = False) -> None:

        # check whether inputs are valid
        self.values, self.coordinates = self.check_coordinates(fieldvalues, coordinates)
        self.coordinate_system = self.check_coordinate_system(coordinates, coordinate_system)
        self.units = self.check_units(coordinates, units)
        self.name = self.check_name(name)
        if vocal:
            print("inputs are valid")
        
        self.shape = self.values.shape
        self.dims = list(self.coordinates.keys())
        self.ndims = len(self.coordinates)
        self.limits = {dim: (self.coordinates[dim][0], self.coordinates[dim][-1]) for dim in self.dims}
        
        if vocal:
            print(self)

    @staticmethod
    def check_coordinates(fieldvalues: np.ndarray, coordinates: dict[str, Sequence[float]]) -> tuple[np.ndarray, dict[str, Sequence[float]]]:

        # check whether number of dimensions match between inputs
        if not len(fieldvalues.shape) == len(coordinates):
            raise ValueError(f"Number of dimensions of inputs don't match:\nlen(fieldvalues.shape) = {len(fieldvalues.shape)};\tlen(coordinates) = {len(coordinates)}.")
        
        # check whether number of points per dimension match between inputs
        if not fieldvalues.shape == tuple(v.size for dim, v in coordinates.items()):
            raise ValueError(f"Number of points per dimension don't match between input variables.")
        
        # check whether dimensions are supported
        for dim in coordinates.keys():
            if dim not in ["x","y","z","r","theta","phi"]:
                raise ValueError(f"Dimension {dim} is not supported.")
        
        return fieldvalues, coordinates

    @staticmethod
    def check_coordinate_system(coordinates: dict[str, Sequence[float]], coordinate_system: str = None) -> str:
        def raise_not_matching_error() -> None:
            raise ValueError(f"Coordinate system {coordinate_system} does not match coordinates {list(coordinates.keys())}.")
        
        # check whether coordinates fit to coordinate system, if specified
        if len(coordinates) == 1 and "z" in coordinates.keys():
            if coordinate_system is None:
                return "cartesian"
            elif coordinate_system in ["cartesian","cylindrical"]:
                return coordinate_system
            else:
                raise_not_matching_error()
        elif np.all([dim in ["x","y","z"] for dim in coordinates.keys()]):
            if coordinate_system is None:
                return "cartesian"
            elif coordinate_system == "cartesian":
                return coordinate_system
            else:
                raise_not_matching_error()
        elif np.all([dim in ["r","phi"] for dim in coordinates.keys()]):
            if coordinate_system is None:
                return "polar"
            elif coordinate_system in ["polar", "cylindrical", "spherical"]:
                return coordinate_system
            else:
                raise_not_matching_error()
        elif np.all([dim in ["r","phi","z"] for dim in coordinates.keys()]):
            if coordinate_system is None:
                return "cylindrical"
            elif coordinate_system == "cylindrical":
                return coordinate_system
            else:
                raise_not_matching_error()
        elif np.all([dim in ["r","theta","phi"] for dim in coordinates.keys()]):
            if coordinate_system is None:
                return "spherical"
            elif coordinate_system == "spherical":
                return coordinate_system
            else:
                raise_not_matching_error()
        else:
            raise ValueError(f"Coordinate set {list(coordinates.keys())} is not supported.")
    
    @staticmethod
    def check_units(coordinates: dict[str, Sequence[float]], units: dict[str, str]) -> dict[str, str]:
        if units is None:
            units = {dim: None for dim in coordinates.keys()}
            units["field"] = None
        else:
            if len(units) != len(coordinates)+1:
                raise ValueError(f"Number of units ({len(units)}) does not match number of dimensions ({len(coordinates)}+1).")
            for dim in coordinates.keys():
                if dim not in units.keys():
                    raise ValueError(f"Unit for dimension {dim} is not specified.")
            if "field" not in units.keys():
                raise ValueError(f"Unit for field is not specified.")
            for unitkey, unitvalue in units.items():
                if unitvalue is not None and not isinstance(unitvalue, str):
                    raise ValueError(f"Unit for {unitkey} is not a string.")
                elif unitvalue is None:
                    units[unitkey] = None
        return units

    @staticmethod
    def check_name(name) -> str:
        if name is not None:
            if isinstance(name, str):
                return name
            else:
                raise ValueError(f"Name is not a string.")
        else:
            return None
            
    def make_newname(self, addstr: str, other: "field" = None, endstr: str = None) -> Union[str,None]:
        if self.name is None or (other is not None and other.name is None):
            return None
        if other is None:
            return self.name + addstr
        else:
            return self.name + addstr + other.name

    def integrate_all_dimensions(self, vocal: bool = False) -> float:
        return self.integrate_dimensions(dims=self.dims, vocal=vocal)
    
    def integrate_dimensions(self, dims: list[str], limits: list[tuple[float, float]] = None, newname: str = None, vocal: bool = False) -> Union["field", float]:

        # check whether lists of dimensions and limits match in length
        if limits is None:
            limits = [(self.coordinates[dim][0], self.coordinates[dim][-1]) for dim in dims]
        elif len(dims) != len(limits):
            raise ValueError(f"Number of dimensions ({len(dims)}) and limits ({len(limits)}) don't match.")
        else:
            for ilim, lim in enumerate(limits):
                if lim==None:
                    limits[ilim] = (self.coordinates[dims[ilim]][0], self.coordinates[dims[ilim]][-1])
        
        integrand = self.values
        mg = np.meshgrid(*[coord for dim, coord in self.coordinates.items()], indexing="ij")

        # # adapting line element for polar/cylindrical/spherical coordinates
        if self.coordinate_system in ["polar", "cylindrical"]:
            if "r" in dims:
                if "phi" in self.dims and "phi" not in dims:
                    raise ValueError(f"Integration over r collapses that dimension, but for a future integration over phi, r is needed. If you want to integrate over both, include both r and phi in dims.")
            if "phi" in dims:
                integrand = integrand*mg[list(self.coordinates.keys()).index("r")] # multiply with r when integrating over phi in polar/cylindrical coordinates
        elif self.coordinate_system == "spherical":
            if "r" in dims:
                if "theta" in self.dims and "theta" not in dims:
                    raise ValueError(f"Integration over r collapses that dimension, but for a future integration over theta, r is needed. If you want to integrate over both, include both r and theta in dims.")
                elif "phi" in self.dims and "phi" not in dims:
                    raise ValueError(f"Integration over r collapses that dimension, but for a future integration over phi, r is needed. If you want to integrate over both, include both r and phi in dims.")
            if "theta" in dims:
                integrand = integrand*mg[list(self.coordinates.keys()).index("r")] * np.sin(mg[list(self.coordinates.keys()).index("theta")]) # multiply with r*sin(theta) when integrating over theta in spherical coordinates
            elif "phi" in dims:
                integrand = integrand*mg[list(self.coordinates.keys()).index("r")] # multiply with r when integrating over phi in spherical coordinates
        elif self.coordinate_system == "cartesian":
            pass
        else:
            raise ValueError(f"Integration of coordinate system {self.coordinate_system} is not supported.")

        # loop over dimensions to integrate
        for idim, (dim, limit) in enumerate(zip(dims,limits)):
            # check whether limits are valid for each dimension
            if dim not in self.coordinates.keys():
                raise ValueError(f"Dimension {dim} is not defined for this field.")
            if limit[0] >= limit[1]:
                raise ValueError(f"Lower limit {limit[0]} is greater than upper limit {limit[1]} for dimension {dim}.")
            if limit[0] < self.coordinates[dim][0] or limit[1] > self.coordinates[dim][-1]:
                raise ValueError(f"Integration limits {limit} are out of bounds for dimension {dim}.")

            # determining limits
            axis = self.dims.index(dim)-idim # which axis to integrate over
            ilim = [np.argmin(np.abs(self.coordinates[dim]-limit[0])), np.argmin(np.abs(self.coordinates[dim]-limit[1]))] # indices of integration limits for that dimension

            if vocal:
                if self.name is not None:
                    ptext = f"{self.name}: "
                else:
                    ptext = ""
                ptext += f"integrating dimension {idim}: {dim} from "
                if self.units[dim] is not None:
                    ptext += f"{self.coordinates[dim][ilim[0]]:g} {self.units[dim]} to {self.coordinates[dim][ilim[1]]:g} {self.units[dim]}"
                else:
                    ptext += f"{self.coordinates[dim][ilim[0]]:g} to {self.coordinates[dim][ilim[1]]:g}"
                print(ptext)

            # actual integration
            integrand = np.trapz(np.take(integrand,range(ilim[0],ilim[1]+1),axis=axis), x=self.coordinates[dim][ilim[0]:ilim[1]+1], axis=axis)
        
        if isinstance(integrand, float):
            return integrand
        else:
            if newname is None:
                newname = self.make_newname("_integrated_" + "_".join([f"{dim}={limit[0]:g}-{limit[1]:g}" for dim, limit in zip(dims,limits)]))
            newcoordinates = self.coordinates.copy()
            [newcoordinates.pop(dim) for dim in dims]
            newunits = self.units.copy()
            [newunits.pop(dim) for dim in dims]
            return field(integrand, newcoordinates, coordinate_system=self.coordinate_system, units=newunits, name=newname, vocal=vocal)
        
    def normalize(self, vocal: bool = False)-> "field":
        newname = self.make_newname("_normalized")
        return field(self.values/self.integrate_all_dimensions(vocal=vocal), self.coordinates, coordinate_system=self.coordinate_system, units=self.units, name=newname, vocal=vocal)
    
    def normalize_abs2(self, vocal: bool = False) -> "field":
        N = (abs(self)**2).integrate_all_dimensions()
        newname = self.make_newname("_normalized_abs2")
        return field(self.values/np.sqrt(N), self.coordinates, coordinate_system=self.coordinate_system, units=self.units, name=newname, vocal=vocal)
    
    def average_dimensions(self, dims: list[str], vocal: bool = False) -> Union["field", float]:
        denominator = np.prod([abs(self.coordinates[dim][-1] - self.coordinates[dim][0]) for dim in dims])
        newname = self.make_newname("_averaged_" + "&".join(dims))
        return self.integrate_dimensions(dims=dims, newname=newname, vocal=vocal)

    def crosscut(self, dim: str, icut: int, newname: str = None) -> "field":
        if dim not in self.coordinates.keys():
            raise ValueError(f"Dimension {dim} is not defined for this field.")
        if icut < 0 or icut >= self.coordinates[dim].size:
            raise ValueError(f"Index {icut} for dimension {dim} is out of bounds.")
        if newname is None:
            addstr = f"_crosscut_{dim}={self.coordinates[dim][icut]:g}"
            if self.units[dim] is not None:
                addstr += f" ({self.units[dim]})"
            newname = self.make_newname(addstr)
        values = self.values.copy()
        coordinates = self.coordinates.copy()
        units = self.units.copy()
        values = np.take(values, icut, axis=self.dims.index(dim))
        del coordinates[dim]
        del units[dim]
        return field(values, coordinates, coordinate_system=self.coordinate_system, units=units, name=newname)
    
    def plot(self, vocal=False, figax: tuple[plt.figure, plt.axes] = None, plotkwargs: dict = {}, cbarkwargs: dict = {}) -> tuple[plt.figure, plt.axes]:
        if self.ndims == 1:
            fig, ax = self.plot_1D(figax=figax, plotkwargs=plotkwargs)
        elif self.ndims == 2:
            fig, ax = self.plot_2D(figax=figax, plotkwargs=plotkwargs, cbarkwargs=cbarkwargs)
        elif self.ndims == 3:
            if vocal:
                printtext = f"Plotting 2D slice of 3D field at {self.dims[2]} = {self.coordinates[self.dims[2]][self.coordinates[self.dims[2]].size//2]}."
                if self.units[self.dims[2]] is not None:
                    printtext += f" ({self.units[self.dims[2]]})"
                print(printtext)
            fig, ax = self.crosscut(self.dims[2], self.coordinates[self.dims[2]].size//2).plot_2D(figax=figax, plotkwargs=plotkwargs, cbarkwargs=cbarkwargs)
        else:
            raise ValueError(f"Plotting of {self.ndims}D fields is not supported.")
        return fig, ax

    def plot_1D(self, figax:  tuple[plt.figure, plt.axes] = None, plotkwargs: dict = {}) -> tuple[plt.figure, plt.axes]:
        if self.ndims != 1:
            raise ValueError(f"Field is not 1D.")

        if figax is None:
            fig, ax = plt.subplots()
        dim = self.dims[0]
        ax.plot(self.coordinates[dim], self.values, **plotkwargs)
        xlabel = dim
        ylabel = "field"
        if self.units[xlabel] is not None:
            xlabel += f" ({self.units[xlabel]})"
        if self.units["field"] is not None:
            ylabel += f" ({self.units['field']})"
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(self.name)

        return fig, ax

    def plot_2D(self, figax:  tuple[plt.figure, plt.axes] = None, plotkwargs: dict = {}, cbarkwargs: dict = {}) -> tuple[plt.figure, plt.axes]:
        if self.ndims != 2:
            raise ValueError(f"Field is not 2D.")
        
        if figax is None:
            fig, ax = plt.subplots()
        x1g, x2g = np.meshgrid(self.coordinates[self.dims[0]], self.coordinates[self.dims[1]], indexing="ij")
        im = ax.pcolormesh(x1g, x2g, self.values, **plotkwargs)
        cbar = fig.colorbar(im, ax=ax, **cbarkwargs)

        # labels
        xlabel, ylabel = self.dims
        cbarlabel = "field"
        if self.units[xlabel] is not None:
            xlabel += f" ({self.units[xlabel]})"
        if self.units[ylabel] is not None:
            ylabel += f" ({self.units[ylabel]})"
        if self.units["field"] is not None:
            cbarlabel += f" ({self.units['field']})"
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(self.name)
        cbar.set_label(cbarlabel)
        
        return fig, ax

    def stitch(self, other: "field", dim: str, newname: str = None) -> "field":
        if not self.coordinate_system == other.coordinate_system:
            raise ValueError(f"Coordinate systems of fields don't match.")
        if dim not in self.coordinates.keys() or dim not in other.coordinates.keys():
            raise ValueError(f"Dimension {dim} is not defined for both fields.")
        for odim in self.coordinates.keys():
            if odim != dim and not np.all(np.vectorize(math.isclose)(self.coordinates[odim], other.coordinates[odim])):
                raise ValueError(f"Coordinates of dimension {odim} don't match between fields.")
        if not self.units == other.units:
            raise ValueError(f"Units of fields don't match.")
        if newname is None:
            newname = self.make_newname("&", other=other, endstr="_stitched_"+f"{dim}")

        values = np.concatenate((self.values, other.values), axis=self.dims.index(dim))
        coordinates = self.coordinates.copy()
        coordinates[dim] = np.append(self.coordinates[dim], other.coordinates[dim])
        print(values.shape, [coordinates[dim].shape for dim in coordinates.keys()])
        return field(values, coordinates, coordinate_system=self.coordinate_system, units=self.units, name=newname)
    
    def split(self, dim: str, icuts: Union[int,list[int]], newname: str = None) -> list["field"]:
        if dim not in self.coordinates.keys():
            raise ValueError(f"Dimension {dim} is not defined for this field.")
        if not isinstance(icuts, list):
            icuts = [icuts]
        for icut in icuts:
            if icut <= 1 or icut >= self.coordinates[dim].size - 1:
                raise ValueError(f"Index {icut} for dimension {dim} is out of bounds to produce cut.") # split results need to have at least 2 points in the dimension that is split
        fields = []
        for kcut in range(len(icuts)+1):
            #TODO: check whether icuts are implemented correctly
            if kcut == 0:
                indices = range(0,icuts[kcut])
            elif kcut == len(icuts):
                indices = range(icuts[kcut-1]+1,self.coordinates[dim].size)
            else:
                indices = range(icuts[kcut-1]+1,icuts[kcut])
            coordinates = self.coordinates.copy()
            coordinates[dim] = self.coordinates[dim][indices]
            values = np.take(self.values, indices, axis=self.dims.index(dim))
            if newname is None:
                newname = self.make_newname("_split_" + f"{dim}+{kcut}")
            fields.append(field(values, coordinates, coordinate_system=self.coordinate_system, units=self.units, name=newname))
        return fields


    def make_interpolator(self) -> callable:
        if self.coordinate_system == "cartesian":
            return RegularGridInterpolator([coord for dim, coord in self.coordinates.items()], self.values)
        else:
            return CloughTocher2DInterpolator(list(zip(*[coord.flat for dim, coord in self.coordinates.items()])), self.values.flat)
    
    def interpolate_to_coordinates(self, coordinates: dict) -> "field":

        # check whether dimensions are supported
        for dim in coordinates.keys():
            if dim not in ["x","y","z","r","theta","phi"]:
                raise ValueError(f"Dimension {dim} is not supported.")
        
        interpolator = self.make_interpolator()
        
        # convert coordinates if needed
        todo_error()

        mg = np.meshgrid(*[coord for dim, coord in coordinates.items()], indexing="ij")
        
        newname = self.make_newname("_interpolated")
        return field(interpolator(mg), coordinates, coordinate_system=self.coordinate_system, units=self.units, name=newname)

    def save(self, filename: str) -> None:
        if filename[-7:] != ".pickle":
            filename += ".pickle"
        with open(filename, "wb") as f:
            pickle.dump(self, f)
    
    @staticmethod
    def load(filename: str) -> "field":
        if filename[-7:] != ".pickle":
            filename += ".pickle"
        with open(filename, "rb") as f:
            return pickle.load(f)
    
    def export_to_csv(self, filename: str = None) -> None:
        if filename is None:
            filename = self.name+".csv"
            print(f"No filename given, exporting field to {filename}.")
        elif filename[-4:] != ".csv":
            filename += ".csv"
        
        mg = np.meshgrid(*[coord for dim, coord in self.coordinates.items()], indexing="ij")
        dfvalues = np.array([*[coord.flatten() for coord in mg], self.values.flatten()]).T
        colnames = list(self.coordinates.keys()) + ["field"]
        for colname in colnames:
            if self.units[colname] is not None:
                colnames[colnames.index(colname)] += f" ({self.units[colname]})"
        df = pd.DataFrame(dfvalues, columns=colnames)
        df.to_csv(filename, index=False)
    
    @staticmethod
    def load_from_csv(filename: str, newname: str = None) -> "field":
        if filename[-4:] != ".csv":
            filename += ".csv"
        df = pd.read_csv(filename)
        coordinates = {colname.split(" (")[0]: np.unique(df[colname].to_numpy()) for colname in df.columns[:-1]}
        units = {colname.split(" (")[0]: colname.split(" (")[1][:-1] for colname in df.columns if "(" in colname}
        values = df[df.columns[-1]].to_numpy().reshape([coordinates[dim].size for dim in coordinates.keys()])
        if newname is None:
            newname = os.path.split(filename)[-1][:-4]
        return field(values, coordinates, units=units, name=newname)

    @check_field_compatibility_strict
    def overlap(self, other: "field", vocal: bool = False) -> float:
        return (self*other.conj()).integrate_all_dimensions(vocal=vocal)
    
    def conj(self) -> "field":
        newname = self.make_newname("_conj")
        return field(self.values.conj(), self.coordinates, coordinate_system=self.coordinate_system, units=self.units, name=newname)

    ######### magic methods #########
    # unary operators    
    def __repr__(self) -> str:
        return f"field in {self.ndims}D, {self.coordinate_system} coordinates {list(self.coordinates.keys())} with units {self.units} and shape {self.shape}"
    
    def __abs__(self) -> "field":
        newname = self.make_newname("_abs")
        return field(np.abs(self.values), self.coordinates, coordinate_system=self.coordinate_system, units=self.units, name=newname)
    
    def __pow__(self, power: float) -> "field":
        newname = self.make_newname(f"_pow{power}")
        return field(np.float_power(self.values,power), self.coordinates, coordinate_system=self.coordinate_system, units=self.units, name=newname)
    
    # binary operators
    @check_field_compatibility_strict
    def __eq__(self, other: "field") -> bool:
        return np.all(self.values == other.values) and np.all(self.coordinates == other.coordinates)
    
    @check_field_compatibility_incl_numbers
    def __add__(self, other: Union["field",int,float,complex]) -> "field":
        if isinstance(other, field):
            newname = self.make_newname("_+_",other)
            return field(self.values+other.values, self.coordinates, coordinate_system=self.coordinate_system, units=self.units, name=newname)
        else:
            newname = self.name
            return field(self.values+other, self.coordinates, coordinate_system=self.coordinate_system, units=self.units, name=newname)
    
    def __radd__(self, other):
        return self.__add__(other)
    
    @check_field_compatibility_incl_numbers
    def __sub__(self, other: Union["field",int,float,complex]) -> "field":
        if isinstance(other, field):
            newname = self.make_newname("_-_",other)
            return field(self.values-other.values, self.coordinates, coordinate_system=self.coordinate_system, units=self.units, name=newname)
        else:
            newname = self.name
            return field(self.values-other, self.coordinates, coordinate_system=self.coordinate_system, units=self.units, name=newname)
    
    def __rsub__(self, other):
        return (self*(-1)).__add__(other)
    
    @check_field_compatibility_incl_numbers
    def __mul__(self, other: Union["field",int,float,complex]) -> "field":
        if isinstance(other, field):
            newname = self.make_newname("_*_",other)
            return field(self.values*other.values, self.coordinates, coordinate_system=self.coordinate_system, units=self.units, name=newname)
        else:
            newname = self.name
            return field(self.values*other, self.coordinates, coordinate_system=self.coordinate_system, units=self.units, name=newname)
    
    def __rmul__(self, other):
        return self.__mul__(other)
    
    @check_field_compatibility_incl_numbers
    def __truediv__(self, other: Union["field",int,float,complex]) -> "field":
        if isinstance(other, field):
            newname = self.make_newname("_/_",other)
            return field(self.values/other.values, self.coordinates, coordinate_system=self.coordinate_system, units=self.units, name=newname)
        else:
            newname = self.name
            return field(self.values/other, self.coordinates, coordinate_system=self.coordinate_system, units=self.units, name=newname)
    
    def __rtruediv__(self, other):
        return (self**-1).__mul__(other)