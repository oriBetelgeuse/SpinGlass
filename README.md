# SpinGlass
**сlass SpinGlass(spliting, domain, initial_state)**
> Class for studying one-dimensional spin glasses described by a one-dimensional scalar field in the "multiplicative pumping + diffusion" model.

**Parametrs:**
>> **spliting: int**
>>> The number of partitions of the domain.

>> **domain: tuple**
>>> Right and left boundaries of the scope of definition of operators.

>> **domain: np.array**
>>> The initial free energy profile.

**Methods**
>> **random_multiplication(func: function, steps=1) -> None**

>> **local_mins() -> int**
