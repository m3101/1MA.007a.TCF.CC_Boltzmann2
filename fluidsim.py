import numpy as np
from typing import Tuple, List

class Simulation:

    directions = np.array([
        [-1,-1],[ 0,-1],[ 1,-1],
        [-1, 0],[ 0, 0],[ 1, 0],
        [-1, 1],[ 0, 1],[ 1, 1]
    ])
    #directions = directions/(((directions**2).sum(axis=1)**.5)+[0,0,0,0,1,0,0,0,0])[:,np.newaxis]

    weights = np.array([
        1/36,1/9,1/36,
        1/9,4/9,1/9,
        1/36,1/9,1/36
    ])

    def __init__(self,shape:Tuple[int,int], viscosity:float,
                 densities:np.ndarray=None,walls:np.ndarray=None,
                 set_vels:list=[]) -> None:
        """A Lattice-Boltzmann Simulation

        Parameters
        ----------
        shape : Tuple[int,int]
            Width and height (respectively) of the 2d lattice.
        viscosity: float
            Viscosity of the fluid.
        densities: np.ndarray of np.float32, optional
            Current density maps. Should have 3 dimensions of shape (9,shape[0],shape[1]).
        walls: np.ndarray of bool, optional
            Map of which points are considered walls.
        set_vels: list of Tuple[np.ndarray,np.ndarray]
            List of tuples containing a mask and a velocity to be set on that mask.
        """
        self.densities = densities if not (densities is None) else np.zeros((9,shape[0],shape[1]))

        #Start at rest
        self.densities[4] = 1
        self.walls = walls if not (walls is None) else np.zeros(shape,dtype=bool)
        self.sources = set_vels

        self.viscosity = viscosity
        self.shape = shape
    def density(self) -> np.ndarray:
        """Density of the current lattice

        Returns
        -------
        np.ndarray of shape self.shape
            Density of the lattice.
        """
        return self.densities.sum(axis=0)
    def directional(self) -> np.ndarray:
        """Directional map

        Returns
        -------
        np.ndarray of shape (9,width,height,2)
        """
        return Simulation.directions[:,np.newaxis,np.newaxis,:].repeat(self.shape[0],axis=1).repeat(self.shape[1],axis=2)
    def velocity(self, density:np.ndarray=None) -> np.ndarray:
        """Velocity field

        Parameters
        ----------
        density : np.ndarray, optional
            Pre-calculated density map. (see Simulation.density).
        Returns
        -------
        np.ndarray of shape (self.shape[0],self.shape[1],2)
            Vector field with the velocities for each point.
        """
        #Directional velocity vectors. (9,width,height,2)
        directional = self.directional()
        #Normalising the density
        normdens = self.densities / self.densities.sum(axis=0)[np.newaxis,:,:]
        #Weighting by density
        directional = normdens[:,:,:,np.newaxis] * directional
        #Summing all directions
        vel = directional.sum(axis=0)
        #Applying sources
        for s in self.sources:
            #self.densities[:,s[0]]/=self.densities.sum(axis=0)[np.newaxis,s[0]]
            vel[s[0]] = s[1]
        return vel
    def equilibrium(self) -> np.ndarray:
        """Redistributes the densities to achieve equilibrium
        
        Returns
        -------
        np.ndarray of shape (9,width,height)
        """
        density = self.density()
        directional = self.directional()
        velocities = self.velocity()
        sqmoduli = (velocities**2).sum(axis=2)
        sqmoduli[sqmoduli>.7]=.7
        ret = self.densities.copy()
        for w,e,i in zip(Simulation.weights,Simulation.directions,range(9)):
            #Scalar product of the velocities by that direction
            sp = (directional[i]*velocities).sum(axis=2)
            sp[sp>.7]=.7
            sp[sp<-.7]=-.7
            #print(i,w,sp.min(),sqmoduli.max())
            ret[i] = density*w*(1+3*sp+sp**2*9/2-sqmoduli*3/2)
        ret[ret<0]=.1
        ret[4,:,:][ret[4,:,:]==0]=.1
        return ret
    def collision(self) -> None:
        """Simulates a collision step
        """
        #This is a workaround
        #For some reason the Boltzmann distribution is
        #yielding negative densities
        #self.densities += self.densities.min(axis=0)[np.newaxis,:,:]
        eq = self.equilibrium()
        self.densities =(1-self.viscosity) * self.densities + self.viscosity * eq
        self.densities[:,0,:] = eq[:,0,:]
        self.densities[:,-1,:] = eq[:,-1,:]
        self.densities[:,:,0] = eq[:,:,0]
        self.densities[:,:,-1] = eq[:,:,-1]
    def streaming(self) -> None:
        """Simulates a streaming step
        """
        #0 1 2
        #3 4 5
        #6 7 8
        temp = np.zeros((9,self.shape[0]+2,self.shape[1]+2))
        temp[:,0,:] = (temp[:,0,:]+1)*Simulation.weights[:,np.newaxis]
        temp[:,-1,:] = (temp[:,-1,:]+1)*Simulation.weights[:,np.newaxis]
        temp[:,:,0] = (temp[:,:,0]+1)*Simulation.weights[:,np.newaxis]
        temp[:,:,-1] = (temp[:,:,-1]+1)*Simulation.weights[:,np.newaxis]
        temp[:,1:-1,1:-1] = self.densities
        #NW
        temp[0][  :-1,  :-1] = temp[0][1:,1:]
        #N
        temp[1][  :  ,  :-1] = temp[1][:,1:]
        #NE
        temp[2][ 1:  ,  :-1] = temp[2][:-1,1:]
        #E
        temp[5][ 1:  ,  :  ] = temp[5][:-1,:]
        #SE
        temp[8][ 1:  , 1:  ] = temp[8][:-1,:-1]
        #S
        temp[7][  :  , 1:  ] = temp[7][:,:-1]
        #SW
        temp[6][  :-1, 1:  ] = temp[6][1:,:-1]
        #W
        temp[3][  :-1,  :  ] = temp[3][1:,:]
        self.densities = temp[:,1:-1,1:-1]        

        #Walls
        self.densities[:,self.walls] = self.densities[[8,7,6,5,4,3,2,1,0]][:,self.walls]
    def step(self) -> None:
        """Steps the simulation one frame forwards (collision and streaming)
        """
        #Collision
        self.collision()
        #Streaming
        self.streaming()

class Smoke:
    def __init__(self, shape:Tuple[int,int], sources:np.ndarray, initial_state:np.ndarray = None) -> None:
        """A density field that can be shaped by a velocity field

        Parameters
        ----------
        shape : Tuple[int,int]
            The shape of the field (width,height).
            Must be the same shape as the velocity field used in simulation steps.
        sources: np.ndarray of bool of shape (width,height)
            Which points will be constant sources of "smoke".
        initial_state: np.ndarray of float of shape (width,height), optional
            Initial state of the field.
        """
        self.field = initial_state.copy() if not (initial_state is None) else np.zeros(shape)
        self.sources = sources.copy()
        self.shape = shape[:]
    def gradient(self,field:np.ndarray = None) -> np.ndarray:
        """Calculates the gradient of the density field

        Parameters
        ----------
        field : np.ndarray of shape (width,height),optional
            An arbitrary field

        Returns
        -------
        np.ndarray of shape (width,height,2)
        """
        field = field if not (field is None) else self.field
        gradient = np.zeros((field.shape[0],field.shape[1],2))
        #X gradient
        gradient[1:-1,:,0] = (field[2:,:]-field[:-2,:])/2
        gradient[0,:,0] = field[1,:]-field[0,:]
        gradient[-1,:,0] = field[-1,:]-field[-2,:]
        #Y gradient
        gradient[:,1:-1,1] = (field[:,2:]-field[:,:-2])/2
        gradient[:,0,1] = field[:,1]-field[:,0]
        gradient[:,-1,1] = field[:,-1]-field[:,-1]
        return gradient
    def divergence(self) -> np.ndarray:
        """Calculates the divergence of the density field

        Returns
        -------
        np.ndarray of shape (width,height)
        """
        grad = self.gradient()
        dx = self.gradient(grad[:,:,0])[:,:,0]
        dy = self.gradient(grad[:,:,1])[:,:,1]
        return dx+dy
    def _alignment(self,velocities:np.ndarray)->np.ndarray:
        grad = self.gradient()
        mod = ((grad**2).sum(axis=2)**.5)[:,:,np.newaxis]
        mod[mod==0] = 1
        grad /= mod

        vmod = ((velocities**2).sum(axis=2)**.5)[:,:,np.newaxis]
        vmod[vmod==0]=1
        velocities = velocities/vmod
        return (grad*velocities).sum(axis=2)
    def step(self, velocities:np.ndarray, sources = True) -> None:
        """Calculates a simulation step

        Parameters
        ----------
        velocities : np.ndarray of shape (width,height,2)
            Velocity vector field
        """
        if sources:
            self.field[self.sources] = 1
        
        gmod = self.gradient()
        gmod = (gmod**2).sum(axis=2)**.5
        mod = (velocities**2).sum(axis=2)**.5 * gmod
        delta = mod*(-self._alignment(velocities))
        self.field+=delta
        self.field[self.field<0]=0