import numpy as np
from scipy.integrate import quad
from abc import ABC


######### decorators #########
def check_field_compatibility(func: callable) -> callable:
    def wrapper(self, other: "field", *args, **kwargs):
        if not self.coordinates == other.coordinates:
            raise ValueError(f"Coordinates of fields don't match!")
        elif not self.shape == other.shape:
            raise ValueError(f"Shapes of fields don't match: {self.shape} and {other.shape}.")
        else:
            return func(self, other, *args, **kwargs)

    return wrapper



class field():

    def __init__(self, fieldvalues: np.ndarray, coordinates: dict, coordinate_system=None, vocal=False) -> None:

        # check whether inputs are valid
        self.check_coordinates(fieldvalues, coordinates)
        self.coordinate_system = self.check_coordinate_system(coordinates, coordinate_system)
        if vocal:
            print("inputs are valid")
        
        self.values = fieldvalues
        self.shape = fieldvalues.shape
        self.coordinates = coordinates
        self.dims = list(coordinates.keys())
        self.ndims = len(coordinates)
        if vocal:
            print(self)

    @staticmethod
    def check_coordinates(fieldvalues: np.ndarray, coordinates: dict) -> None:

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

    @staticmethod
    def check_coordinate_system(coordinates: dict, coordinate_system=None) -> str:
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

    def integrate_all_dimensions(self, vocal: bool = False) -> float:
        return self.integrate_dimensions(dims=self.dims, vocal=vocal)
    
    def integrate_dimensions(self, dims: list[str], limits: list[tuple[float, float]] = None, vocal: bool = False) -> "field":

        # check whether lists of dimensions and limits match in length
        if limits is None:
            limits = [(self.coordinates[dim][0], self.coordinates[dim][-1]) for dim in dims]
        elif len(dims) != len(limits):
            raise ValueError(f"Number of dimensions ({len(dims)}) and limits ({len(limits)}) don't match.")
        
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

            if vocal:
                print(f"integrating dimension {idim}: {dim} from {limit[0]} to {limit[1]}")

            # actual integration
            axis = self.dims.index(dim)-idim # which axis to integrate over
            ilim = [np.argmin(np.abs(self.coordinates[dim]-limit[0])), np.argmin(np.abs(self.coordinates[dim]-limit[1]))] # indices of integration limits for that dimension
            indices = [slice(None)]*(self.ndims - idim)
            indices[axis] = slice(ilim[0],ilim[1]+1)
            integrand = np.trapz(integrand[tuple(indices)], x=self.coordinates[dim][ilim[0]:ilim[1]+1], axis=axis)
        
        if isinstance(integrand, float):
            return integrand
        else:
            return field(integrand, {dim: self.coordinates[dim] for dim in self.dims if dim not in dims}, coordinate_system=self.coordinate_system, vocal=vocal)
            

    
    def normalize(self, vocal: bool = False)-> "field":
        return field(self.values/self.integrate_all_dimensions(vocal=vocal), self.coordinates)
    
    def normalize_abs2(self, vocal: bool = False) -> "field":
        N = field(self.values*self.values.conj(), self.coordinates).integrate_all_dimensions(vocal=vocal)
        # N = (self*self.conj()).integrate_all_dimensions()
        return field(self.values/np.sqrt(N), self.coordinates)
    
    @check_field_compatibility
    def overlap(self, other: "field", vocal: bool = False) -> float:
        return (self*other.conj()).integrate_all_dimensions(vocal=vocal)
    
    def conj(self) -> "field":
        return field(self.values.conj(), self.coordinates)

    ######### magic methods #########
    # unary operators    
    def __repr__(self) -> str:
        return f"field in {self.ndims}D, {self.coordinate_system} coordinates {list(self.coordinates.keys())} with shape {self.shape}"
    
    def __abs__(self) -> "field":
        return field(np.abs(self.values), self.coordinates)
    
    def __pow__(self, power: float) -> "field":
        return field(self.values**power, self.coordinates)
    
    # binary operators
    @check_field_compatibility
    def __eq__(self, other: "field") -> bool:
        return np.all(self.values == other.values) and np.all(self.coordinates == other.coordinates)
    
    @check_field_compatibility
    def __add__(self, other: "field") -> "field":
        return field(self.values+other.values, self.coordinates)
    
    @check_field_compatibility
    def __sub__(self, other: "field") -> "field":
        return field(self.values-other.values, self.coordinates)
    
    @check_field_compatibility
    def __mul__(self, other: "field") -> "field":
        return field(self.values*other.values, self.coordinates)

    @check_field_compatibility
    def __truediv__(self, other: "field") -> "field":
        return field(self.values/other.values, self.coordinates)


    
    