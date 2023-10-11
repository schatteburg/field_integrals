import numpy as np
from scipy.integrate import quad
from abc import ABC


######### decorators #########
def checked_field_compatibility(func: callable) -> callable:
    def wrapper(self, other: "field", *args, **kwargs):
        if not self.coordinate_system == other.coordinate_system:
            raise ValueError(f"Coordinate systems of fields don't match: {self.coordinate_system} and {other.coordinate_system}.")
        elif not self.shape == other.shape:
            raise ValueError(f"Shapes of fields don't match: {self.shape} and {other.shape}.")
        else:
            return func(self, other, *args, **kwargs)

    return wrapper



class field():

    def __init__(self, fieldvalues: np.ndarray, coordinates: list[np.ndarray], coordinate_system: str) -> None:
        if coordinate_system in ["polar", "cylindrical", "spherical"]:
            print("WARNING: for polar/cylindrical/spherical coordinate_systems, coordinates have to follow [r, (theta), phi, (z)] order for correct integration!")
        # if not (type(fieldvalues) == np.ndarray and type(coordinates)==)


        # check whether number of dimensions match between inputs
        if not len(fieldvalues.shape) == len(coordinates):
            raise ValueError(f"Number of dimensions of inputs don't match:\nlen(fieldvalues.shape) = {len(fieldvalues.shape)};\tlen(coordinates) = {len(coordinates)}.")
        else:
            self.ndims = len(coordinates)
        
        # check whether number of points per dimension match between inputs
        if not fieldvalues.shape == tuple(v.size for v in coordinates):
            raise ValueError(f"Number of points per dimension don't match between input variables.")
        else:
            self.shape = fieldvalues.shape
        
        # check whether number of dimensions and coordinate system are supported
        if self.ndims == 1 and coordinate_system not in ["cartesian", "polar"]  or self.ndims == 2 and coordinate_system not in ["cartesian", "polar"] or self.ndims == 3 and coordinate_system not in ["cartesian", "cylindrical","spherical"] or self.ndims > 3 or self.ndims <=0:
            raise ValueError(f'Number of dimensions ({self.ndims}) and/or coordinate system ({coordinate_system}) are not supported.')

        self.values = fieldvalues
        self.coordinates = coordinates
        self.coordinate_system = coordinate_system
    
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
        res = self.integrate_dimensions(dims=list(range(self.ndims)), vocal=vocal)
        if type(res) == field:
            return res.values
        else:
            return res
    
    def integrate_dimensions(self, dims: list[int], limits: Optional[list[tuple[float, float]]] = None, vocal: bool = False) -> "field":
        # check whether lists of dimensions and limits match in length
        if limits is None:
            limits = [(self.coordinates[dim][0], self.coordinates[dim][-1]) for dim in dims]
        elif len(dims) != len(limits):
            raise ValueError(f"Number of dimensions ({len(dims)}) and limits ({len(limits)}) don't match.")
        
        integrand = self.values
        for dim, limit in zip(dims,limits):
            # check whether limits are valid for each dimension
            if dim >= self.ndims:
                raise ValueError(f"Dimension {dim} is out of bounds for field with {self.ndims} dimensions.")
            if limit[0] >= limit[1]:
                raise ValueError(f"Lower limit {limit[0]} is greater than upper limit {limit[1]} for dimension {dim}.")
            if limit[0] < self.coordinates[dim][0] or limit[1] > self.coordinates[dim][-1]:
                raise ValueError(f"Integration limits {limit} are out of bounds for dimension {dim}.")

            if vocal:
                print(f"integrating dimension {dim} from {limit[0]} to {limit[1]}")

            # adapting line element for polar/cylindrical/spherical coordinates
            if self.coordinate_system in ["polar","cylindrical","spherical"] and dim == 1:
                integrand = integrand*self.coordinates[0] # multiply with r when integrating over phi/theta
            elif self.coordinate_system == "spherical" and dim == 2:
                integrand = integrand*self.coordinates[0]*np.sin(self.coordinates[1]) # multiply with r*sin(theta) when integrating over phi

            # actual integration
            ilim = [np.argmin(np.abs(self.coordinates[dim]-limit[0])), np.argmin(np.abs(self.coordinates[dim]-limit[1]))]
            integrand = np.trapz(integrand[ilim[0]:ilim[1]+1], x=self.coordinates[dim][ilim[0]:ilim[1]+1], axis=dim)
        
        if len(dims) == self.ndims:
            return integrand
        else:
            return field(integrand, [self.coordinates[dim] for dim in range(self.ndims) if dim not in dims], self.coordinate_system)
            

    
    def normalize(self, vocal: bool = False)-> "field":
        return field(self.values/self.integrate_all_dimensions(vocal=vocal), self.coordinates, self.coordinate_system)
    
    def normalize_abs2(self, vocal: bool = False) -> "field":
        N = field(self.values*self.values.conj(), self.coordinates, self.coordinate_system).integrate_all_dimensions(vocal=vocal)
        # N = (self*self.conj()).integrate_all_dimensions()
        return field(self.values/np.sqrt(N), self.coordinates, self.coordinate_system)
    
    @checked_field_compatibility
    def overlap(self, other: "field", vocal: bool = False) -> float:
        return (self*other.conj()).integrate_all_dimensions(vocal=vocal)
    
    def conj(self) -> "field":
        return field(self.values.conj(), self.coordinates, self.coordinate_system)

    ######### magic methods #########
    # unary operators    
    def __repr__(self) -> str:
        return f"field in {self.ndims}D, {self.coordinate_system} coordinate system, with shape {self.shape}"
    
    def __abs__(self) -> "field":
        return field(np.abs(self.values), self.coordinates, self.coordinate_system)
    
    def __pow__(self, power: float) -> "field":
        return field(self.values**power, self.coordinates, self.coordinate_system)
    
    # binary operators
    @checked_field_compatibility
    def __eq__(self, other: "field") -> bool:
        return np.all(self.values == other.values) and np.all(self.coordinates == other.coordinates) and self.coordinate_system == other.coordinate_system
    
    @checked_field_compatibility
    def __add__(self, other: "field") -> "field":
        return field(self.values+other.values, self.coordinates, self.coordinate_system)
    
    @checked_field_compatibility
    def __sub__(self, other: "field") -> "field":
        return field(self.values-other.values, self.coordinates, self.coordinate_system)
    
    @checked_field_compatibility
    def __mul__(self, other: "field") -> "field":
        return field(self.values*other.values, self.coordinates, self.coordinate_system)

    @checked_field_compatibility
    def __truediv__(self, other: "field") -> "field":
        return field(self.values/other.values, self.coordinates, self.coordinate_system)


    
    