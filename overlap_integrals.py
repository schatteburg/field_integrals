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
            elif coordinate_system not in ["cartesian","cylindrical"]:
                raise_not_matching_error()
        elif np.all([dim in ["x","y","z"] for dim in coordinates.keys()]):
            if coordinate_system is None:
                return "cartesian"
            elif coordinate_system != "cartesian":
                raise_not_matching_error()
        elif np.all([dim in ["r","phi"] for dim in coordinates.keys()]):
            if coordinate_system is None:
                return "polar"
            elif coordinate_system not in ["polar", "cylindrical", "spherical"]:
                raise_not_matching_error()
        elif np.all([dim in ["r","phi","z"] for dim in coordinates.keys()]):
            if coordinate_system is None:
                return "cylindrical"
            elif coordinate_system != "cylindrical":
                raise_not_matching_error()
        elif np.all([dim in ["r","theta","phi"] for dim in coordinates.keys()]):
            if coordinate_system is None:
                return "spherical"
            elif coordinate_system != "spherical":
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
        # if self.coordinate_system == "cartesian":
        #     integrand = self.values
        # # adapting line element for polar/cylindrical/spherical coordinates
        # elif self.coordinate_system == "polar":
        #     rg, phig = np.meshgrid(self.coordinates["r"], self.coordinates["phi"], indexing="ij")
        #     integrand = self.values*rg
        # elif self.coordinate_system == "cylindrical":
        #     rg, phig, zg = np.meshgrid(self.coordinates["r"], self.coordinates["phi"], self.coordinates["z"], indexing="ij")
        #     integrand = self.values*rg
        # elif self.coordinate_system == "spherical":
        #     rg, thetag, phig = np.meshgrid(self.coordinates["r"], self.coordinates["theta"], self.coordinates["phi"], indexing="ij")
        #     integrand = self.values*rg*rg*np.sin(thetag)
        # else:
        #     raise ValueError(f"Coordinate system {self.coordinate_system} is not supported.")

        mg = np.meshgrid(*[coord for dim, coord in self.coordinates.items()], indexing="ij")

        # # # adapting line element for polar/cylindrical/spherical coordinates
        # if self.coordinate_system in ["polar", "cylindrical"]:
        #     if "r" in dims:
        #         integrand = integrand*mg[list(self.coordinates.keys()).index("r")]
        # elif self.coordinate_system == "spherical":
        #     if "r" in dims and "theta" in dims":
        #         integrand = integrand*mg[list(self.coordinates.keys()).index("r")]**2 * np.sin(mg[list(self.coordinates.keys()).index("theta")]) # multiply with r^2*sin(theta) when integrating over r and theta in spherical coordinates
        #     elif "r" in dims:
        #         integrand = integrand*mg[list(self.coordinates.keys()).index("r")] # multiply with r when integrating over r in spherical coordinates
        #     elif "theta" in dims:
        #         integrand = integrand*np.sin(mg[list(self.coordinates.keys()).index("theta")])
        # elif self.coordinate_system == "cartesian":
        #     pass
        # else:
        #     raise ValueError(f"Coordinate system {self.coordinate_system} is not supported.")

        if "r" in dims:
            if self.coordinate_system == "spherical":
                integrand = integrand*mg[list(self.coordinates.keys()).index("r")]**2 # multiply with r^2 when integrating over r in spherical coordinates
            else:
                integrand = integrand*mg[list(self.coordinates.keys()).index("r")] # multiply with r when integrating over r in polar/cylindrical coordinates
        if "theta" in dims:
            integrand = integrand*np.sin(mg[list(self.coordinates.keys()).index("theta")]) # multiply with r when integrating over theta

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

            # if dim == "r":
            #     rg, *_ = np.meshgrid(*[coord for dim, coord in self.coordinates.items()], indexing="ij")
            #     if self.coordinate_system == "spherical":
            #         integrand = integrand*rg**2 # multiply with r^2 when integrating over r in spherical coordinates
            #     else:
            #         integrand = integrand*rg # multiply with r when integrating over r in polar/cylindrical coordinates
            # elif dim == "theta":
            #     rg, thetag, phig = np.meshgrid(self.coordinates["r"], self.coordinates["theta"], self.coordinates["phi"], indexing="ij")
            #     integrand = integrand*np.sin(thetag) # multiply with r when integrating over theta
                

            # actual integration
            ilim = [np.argmin(np.abs(self.coordinates[dim]-limit[0])), np.argmin(np.abs(self.coordinates[dim]-limit[1]))]
            integrand = np.trapz(integrand[ilim[0]:ilim[1]+1], x=self.coordinates[dim][ilim[0]:ilim[1]+1], axis=self.dims.index(dim)-idim)
        
        if isinstance(integrand, float):
            return integrand
        else:
            if self.coordinate_system == "spherical" and "theta" in dims:
                new_coodinate_system = "spherical"
            else:
                new_coodinate_system = None
            return field(integrand, {dim: self.coordinates[dim] for dim in self.dims if dim not in dims}, coordinate_system=new_coodinate_system, vocal=vocal)
            

    
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
        return f"field in {self.ndims}D, {self.coordinate_system} coordinates {list(self.coordinates.keys())} with shape {self.shape}"
    
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


    
    