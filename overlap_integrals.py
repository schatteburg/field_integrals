import numpy as np
from scipy.integrate import quad
from abc import ABC


######### decorators #########
def checked_field_compatibility(func: callable) -> callable:
    def wrapper(self, other: "field", *args, **kwargs):
        if not self.coordinates == other.coordinates:
            raise ValueError(f"Coordinates of fields don't match!")
        elif not self.shape == other.shape:
            raise ValueError(f"Shapes of fields don't match: {self.shape} and {other.shape}.")
        else:
            return func(self, other, *args, **kwargs)

    return wrapper



class field():

    def __init__(self, fieldvalues: np.ndarray, coordinates: dict) -> None:
        # if not (type(fieldvalues) == np.ndarray and type(coordinates)==)


        # check whether number of dimensions match between inputs
        if not len(fieldvalues.shape) == len(coordinates):
            raise ValueError(f"Number of dimensions of inputs don't match:\nlen(fieldvalues.shape) = {len(fieldvalues.shape)};\tlen(coordinates) = {len(coordinates)}.")
        else:
            self.ndims = len(coordinates)
        
        # check whether number of points per dimension match between inputs
        if not fieldvalues.shape == tuple(v.size for dim, v in coordinates.items()):
            raise ValueError(f"Number of points per dimension don't match between input variables.")
        else:
            self.shape = fieldvalues.shape
        
        # check whether number of dimensions are supported
        for dim in coordinates.keys():
            if dim not in ["x","y","z","r","theta","phi"]:
                raise ValueError(f"Dimension {dim} is not supported.")
        if np.all([dim in ["x","y","z"] for dim in coordinates.keys()]):
            self.coordinate_system = "cartesian"
        elif set(["r","phi"]) == set(coordinates.keys()):
            self.coordinate_system = "polar"
        elif set(["r","phi","z"]) == set(coordinates.keys()):
            self.coordinate_system = "cylindrical"
        elif set(["r","theta","phi"]) == set(coordinates.keys()):
            self.coordinate_system = "spherical"
        else:
            raise ValueError(f"Coordinate set {list(coordinates.keys())} is not supported.")
        
        self.values = fieldvalues
        self.coordinates = coordinates
        self.dims = list(coordinates.keys())
    
    # def integrate_all_dimensions(self, vocal: bool = False) -> float:
    #     if self.ndims == 1:
    #         if vocal:
    #             print('integrating dimension 0')
    #         if self.coordinate_system == "cartesian":
    #             return np.trapz(self.values, x=self.coordinates[0], axis=0)
    #         elif self.coordinate_system in ["polar", "cylindrical"]:
    #             # raise NotImplementedError(f"Coordinate system {self.coordinate_system} is not implemented.")
    #             return np.trapz(self.coordinates[0]*self.values, x=self.coordinates[0], axis=0)
    #     else:
    #         if vocal:
    #             print(f'integrating dimension {self.ndims-1}')
    #         newvalues = np.trapz(self.values, x=self.coordinates[-1], axis=self.ndims-1)
    #         if self.ndims == 3 and self.coordinate_system == "cylindrical":
    #             newcs = "polar"
    #         else:
    #             newcs = self.coordinate_system
    #         return field(newvalues, self.coordinates[:-1], newcs).integrate_all_dimensions(vocal=vocal) # recursive call

    def integrate_all_dimensions(self, vocal: bool = False) -> float:
        return self.integrate_dimensions(dims=self.dims, vocal=vocal)
    
    def integrate_dimensions(self, dims: list[str], limits: list[tuple[float, float]] = None, vocal: bool = False) -> "field":
        # check whether lists of dimensions and limits match in length
        if limits is None:
            limits = [(self.coordinates[dim][0], self.coordinates[dim][-1]) for dim in dims]
        elif len(dims) != len(limits):
            raise ValueError(f"Number of dimensions ({len(dims)}) and limits ({len(limits)}) don't match.")
        
        integrand = self.values
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

            # adapting line element for polar/cylindrical/spherical coordinates
            if dim == "phi":
                if self.coordinate_system == "spherical":
                    integrand = integrand*self.coordinates["r"]*np.sin(self.coordinates["theta"]) # multiply with r*sin(theta) when integrating over phi in spherical coordinates
                else:
                    integrand = integrand*self.coordinates["r"] # multiply with r when integrating over phi in polar/cylindrical coordinates
            elif dim == "theta":
                integrand = integrand*self.coordinates["r"] # multiply with r when integrating over theta
                

            # actual integration
            ilim = [np.argmin(np.abs(self.coordinates[dim]-limit[0])), np.argmin(np.abs(self.coordinates[dim]-limit[1]))]
            integrand = np.trapz(integrand[ilim[0]:ilim[1]+1], x=self.coordinates[dim][ilim[0]:ilim[1]+1], axis=0)
        
        if isinstance(integrand, float):
            return integrand
        else:
            return field(integrand, [self.coordinates[dim] for dim in self.dims if dim not in dims])
            

    
    def normalize(self, vocal: bool = False)-> "field":
        return field(self.values/self.integrate_all_dimensions(vocal=vocal), self.coordinates)
    
    def normalize_abs2(self, vocal: bool = False) -> "field":
        N = field(self.values*self.values.conj(), self.coordinates).integrate_all_dimensions(vocal=vocal)
        # N = (self*self.conj()).integrate_all_dimensions()
        return field(self.values/np.sqrt(N), self.coordinates)
    
    @checked_field_compatibility
    def overlap(self, other: "field", vocal: bool = False) -> float:
        return (self*other.conj()).integrate_all_dimensions(vocal=vocal)
    
    def conj(self) -> "field":
        return field(self.values.conj(), self.coordinates)

    ######### magic methods #########
    # unary operators    
    def __repr__(self) -> str:
        return f"field in {self.ndims}D, coordinates {list(self.coordinates.keys())}, with shape {self.shape}"
    
    def __abs__(self) -> "field":
        return field(np.abs(self.values), self.coordinates)
    
    def __pow__(self, power: float) -> "field":
        return field(self.values**power, self.coordinates)
    
    # binary operators
    @checked_field_compatibility
    def __eq__(self, other: "field") -> bool:
        return np.all(self.values == other.values) and np.all(self.coordinates == other.coordinates)
    
    @checked_field_compatibility
    def __add__(self, other: "field") -> "field":
        return field(self.values+other.values, self.coordinates)
    
    @checked_field_compatibility
    def __sub__(self, other: "field") -> "field":
        return field(self.values-other.values, self.coordinates)
    
    @checked_field_compatibility
    def __mul__(self, other: "field") -> "field":
        return field(self.values*other.values, self.coordinates)

    @checked_field_compatibility
    def __truediv__(self, other: "field") -> "field":
        return field(self.values/other.values, self.coordinates)


    
    