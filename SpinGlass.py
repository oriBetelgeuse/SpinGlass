from scipy import integrate, signal
import numpy as np


class SpinGlass:

    def __init__(self, splitting, domain, initial_state):
        self.x = np.linspace(domain[0], domain[1], splitting)
        self.state = initial_state

    def _random_phase(self, size=None):
        return np.random.uniform(self.x[0], self.x[-1], size=size)

    def _multiplication(self, func):
        vector_func = np.vectorize(func)
        return vector_func(self.x) * self.state

    @staticmethod
    def _diffusion(arg, time, diffusion_coef):
        return 1 / np.sqrt(4 * np.pi * time * diffusion_coef) * np.exp(- arg ** 2 / (4 * diffusion_coef * time))

    def _evolution(self, arg, time, diffusion_coef):
        vector_diffusion = np.vectorize(self._diffusion)
        return integrate.simps(vector_diffusion(self.x - arg, time, diffusion_coef), self.x)

    def random_multiplication(self, func, steps=1):
        vector_func = np.vectorize(func)
        self.state += sum(vector_func(self.x + phase) for phase in self._random_phase(steps))

    def diffusion_evolution(self, time, diffusion_coef):
        vector_evolution = np.vectorize(self._evolution)
        self.state = vector_evolution(self.x, time, diffusion_coef)

    def local_mins(self):
        return signal.argrelmin(self.state)[0].size
