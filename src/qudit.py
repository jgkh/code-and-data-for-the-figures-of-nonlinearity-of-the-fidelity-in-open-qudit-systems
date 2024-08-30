import qutip as qt
import numpy as np
import qutip.operators as op

import matplotlib.pyplot as plt

class UnitOperator(qt.Qobj):  #This creates a class that has the same features as qt.Qobj which are data, dimensions, shape, hermitivity, type.
    
    def __init__(self, s): #This is the definition of the features of UniOperator
        qt.Qobj.__init__(self, s) #because of this line
        
    def __mul__(self, other): #This multiplies the unit operator and other object together.
        if isinstance(other,Qudit): #Checks that the other's class is a Qudit
            return (qt.Qobj.__mul__(self,other)).unit() #Makes a type of product and normalize it with ".unit()" 
        elif isinstance(other,UnitOperator):  #Checks if the other is a Unit Operator then it doesn't have the .unit()
            return UnitOperator(qt.Qobj.__mul__(self,other))  #Simply multiply both operators
        else:
            return qt.Qobj.__mul__(self,other) #If it is neither a qudit or a unitary operator it does this product only which is still unclear. 
        
    def __pow__(self, other):
        return UnitOperator(qt.Qobj.__pow__(self,other)) 
    
    

class Qudit(UnitOperator):  #This creates a subclasss that has the same features of UnitOperator/Qobject. 
    def __init__(self, **kwargs):#d=2,state=None): #This says that this class has an undeterminated number of arguments. 
        
        d = kwargs.get("d", None) #From the dictionary of arguments and we take the value assigned to d. If "d" doesn't appear then assign "None" 
        self.s = kwargs.get("s", None)  #If "d" is not given, then we get s from the list of arguments

        if d: #If d exist
            self.s = (d-1)/2.0  #If the dimension is d we can get which is the spin of the system            
        else:            
            d = int(2*self.s+1) #With the given spin we can obtain the dimension         

        self.dimension = d #Assign the dimension to the class

        self.basis = [qt.basis(d,i) for i in range(d)]  #Generates an array of the vectors of the cannonical basis

        self.offset = kwargs.get("offset", 0) #Offset is like a way to number the states I think
        
        for i in range(self.offset, d+self.offset): #Not sure about this part
            exec("self.q"+str(i)+"=self.basis[i-self.offset]") #What is self.q?
        
        self.sigmas=np.empty((d,d), dtype=qt.Qobj)  #Define a square matrix of the dimension of the system which is qt.Qobj
        for i in range(self.offset, d+self.offset): #Not sure about this part
            for k in range(self.offset, d+self.offset):
                exec("self.s"+str(i)+str(k)+"=self.q"+str(i)+"*self.q"+str(k)+".dag()") #Filling the sigmas somehow. 
                exec("self.sigmas["+str(i-self.offset)+","+str(k-self.offset)+"]=self.s"+str(i)+str(k))
            
        self.state = qt.Qobj(kwargs.get("state",self.basis[d-1])) #What does self.basis[d-1] do?
        if self.state.type=='bra': self.state = self.state.dag()  #If the state is a bra, conver it to a ket. 
        
        self.state = self.state/self.state.norm() #Normalize the state
        
        UnitOperator.__init__(self,self.state)  #Not sure what this does.
        
        self.dmx = self.state * self.state.trans()  #Create the density matrix
        
        self.spacings = kwargs.get("spacings",np.zeros(d))  #These are the energy spacing between the energy levels
        self.H0 = qt.Qobj(np.diag(self.spacings)) #The Hamiltonian in the case of free energy in diagonal form.
        
        self.pulses = []  #An array that contains the pulses. This I think it is done to write the time dependent Hamiltonian
        self.collapse = []  #Probably these are the noise operators

        self.transition_matrix = kwargs.get("transition_matrix", None) #This is the transition matrix, which is a matrix of zeros of dimension dxd

        if self.transition_matrix is None:
            self.transition_matrix = Qudit.ladder_system_adjacency_matrix(d) #If the transition matrix is not given, then we create a ladder system

        if Qudit.is_transition_graph_connected(self.transition_matrix):
            pass

        pass

    def create_qudit_ghz_density_matrix(self):

        psi_zero = self.basis[0]

        psi_d_minus_one = self.basis[-1]

        psi_ghz = (psi_zero + psi_d_minus_one) / np.sqrt(2)

        rho_ghz = psi_ghz * psi_ghz.dag()

        return rho_ghz
        
    def get_j(self,*args):  #Function to get j, which is the total angular momentum? 
        return op.jmat(self.s,*args)  #From QuTip take the operators package and take jmat (not sure what it is)
        
    def get_j_unit(self,*args): #Use the angular momentum to create the UnitOperator
        return UnitOperator(self.get_j(*args))
            
    def get_dm(self): #This is a function to access the density matrix
        return self.dmx
    
    def set_state(self,state):  #Assign the "state" value to the state feature of the UnitOperator class
        self.state=state
        
    def get_spacings(self): #Function to get the spacings
        return self.spacings
    
    def get_H0(self): #Function to get the matrix form of the free Hamiltonian
        return self.H0
    
    def get_sigmas(self): #Function to get the created sigmas
        return self.sigmas
    
    def add_pulse(self,mxel,pulse): #Not sure about this add_pulse function
        if not isinstance(mxel,qt.Qobj):  #Check if mxel is NOT a instance of qt.Qobj
            if isinstance(mxel,str) or isinstance(mxel,int): #Checks if mxel is either a string or an integer
                mxel=eval("self.s"+str(mxel)) #I think this This joints the numbers inside the mxel, so if s=1 and mxel=3 then we should get 13?
                
        self.pulses.append([mxel,lambda t, args: pulse(t)]) #why there is a space between lambda and t? What are we adding to the list of pulses?
        self.pulses.append([mxel.dag(),lambda t, args: np.conjugate(pulse(t))]) #Why are we adding this second list?

    def remove_all_pulses(self):
        self.pulses.clear()
  
    def add_pulseargs(self,mxel,pulse): #Not sure of the goal of this line
        self.pulses.append([mxel,lambda t, args: pulse(t,args)])
        self.pulses.append([mxel.dag(),lambda t, args: np.conjugate(pulse(t,args))])
    
    def add_pulseRWA(self,ti,dur,RWAmx):  #Pulse in the Rotating Wave approximation
        for i in range(0,RWAmx.shape[0]): 
            if RWAmx[i,i]:  
              mxel = self.sigmas[i,i] #mxel becomes equal to the entry of the diagonal of the sigmas. Why?
              Om = RWAmx[i,i]/2 #This seems to be the amplitude of the pulse
              signal = constpulse(ti, dur, Om)  #The constpulse is the function that creates a constant pulse
              self.add_pulse(mxel, signal)

        for i in range(0,RWAmx.shape[0]): 
            for j in range(i+1,RWAmx.shape[0]): #Takes only the entry that are just above the diagonal
                if RWAmx[i,j]:  
                    mxel = self.sigmas[i,j]  
                    Om = RWAmx[i,j] 
                    signal = constpulse(ti, dur, Om)  
                    self.add_pulse(mxel, signal)

    def get_multi_rotating_frame_generator(self, S_adj, om, t, dim):    
        d=dim
        pauli_zs=[Hz(n,m,d) for (n,m) in S_adj]
        M=np.empty((len(S_adj),len(S_adj)))

        for i, (n,nd) in enumerate(S_adj):
            for j, (m,md) in enumerate(S_adj):
                if n==m and nd==md:
                    M[i,j]=2
                elif (m==n and nd!=md) or (m!=n and nd==md):
                    M[i,j]=1
                elif (m==nd or n==md):
                    M[i,j]=-1
                else:
                    M[i,j]=0

        b=[t*om[i] for i in range(len(S_adj))]
        solution=np.linalg.solve(M,b)
        R=np.sum([pauli_zs[i]*solution[i] for i in range(len(S_adj))], axis=0)
        return R
        
    def RWA_to_lab_frame(self, H_RWA, S_adj, om,t, dim, real=True):
        print("Passed first step")
        R=self.get_multi_rotating_frame_generator(S_adj, om,t, dim)
        print("Passed second step")
        H_lab=expm(1j*R)@(H_RWA-R/t)@expm(-1j*R)
        print("Passed third step")
        if real:
            print("Passed fourth step")
            H_lab=np.real(H_lab)
            print("Passed fifth step")
        return qt.Qobj(H_lab)
                         
    def convert_RWAmatrix(self,ti,dur,RWAmx): 
        deltas = [] 
        for i in range(0,RWAmx.shape[0]):
            deltas.append(RWAmx[i,i]) 
        for i in range(0,RWAmx.shape[0]): 
            for j in range(i+1,RWAmx.shape[0]):
                if RWAmx[i,j]:
                    mxel = self.sigmas[i,j]
                    Om = RWAmx[i,j]
                    w = self.spacings[j]-self.spacings[i]+deltas[i]-deltas[j] #Detuning?
                    signal = exppulse(ti, dur, w, Om) 
                    self.add_pulse(mxel, signal)
        
    def convert_RWAmatrix_real(self,ti,dur,RWAmx):
        deltas = []
        for i in range(0,RWAmx.shape[0]):
            deltas.append(RWAmx[i,i])
        for i in range(0,RWAmx.shape[0]):
            for j in range(i+1,RWAmx.shape[0]):
                if RWAmx[i,j]:
                    mxel = self.sigmas[i,j]
                    Om = RWAmx[i,j]*2
                    w = self.spacings[j]-self.spacings[i]+deltas[i]-deltas[j]
                    signal = trigpulse(ti, dur, w, Om)
                    self.add_pulse(mxel, signal)
                    
    def add_coll_op(self,op):
        self.collapse.append(op)

    def remove_coll_op(self, op):
        self.collapse.remove(op)
    
    def evolve(self, t):
        self.H = [self.H0] + self.pulses  #This generates the Hamiltonian with the free and pulse
        
        self.evol = qt.mesolve(self.H, self.dmx, t, self.collapse)  #This uses the solver from QuTip and yields us the expected value of the collapse operators
        return self.evol
    
    def get_evol(self,proj):   
        try:
            return qt.expect(proj,self.evol.states)
        except:
            print("You didn't evolve the qu4it in time.")
    
    def plotpop(self, t):
        plt.figure()
        
        for i in range(int(2*self.s+1)):
            plt.plot(t,self.get_evol(self.sigmas[i,i])) #What are the sigmas exactly?
            
    def plotcoh(self, t):
        plt.figure()
        
        for i in range(int(2*self.s+1)):
           for j in range(i+1,int(2*self.s+1)):
              w=self.spacings[j]-self.spacings[i]
              plt.plot(t,abs(self.get_evol(self.sigmas[i,j]))) #If we have only transitions 12,23,34, and so on there is only a few evolutions plotted?
            
        plt.show()

        plt.figure()
        
        for i in range(int(2*self.s+1)):
           for j in range(i+1,int(2*self.s+1)):
              w = self.spacings[j]-self.spacings[i]
              plt.plot(t,(self.get_evol(self.sigmas[i,j])*np.exp(1.0j*w*t)).real) #Why we have this with an exponential with the spacings?
            
        plt.show()

        plt.figure()
        
        for i in range(int(2*self.s+1)):
           for j in range(i+1,int(2*self.s+1)):
              w = self.spacings[j]-self.spacings[i]
              plt.plot(t,(self.get_evol(self.sigmas[i,j])*np.exp(1.0j*w*t)).imag)
            
        plt.show()
                
    def evolveRWA(self, t):
        options_mesolve = qt.Options()
        options_mesolve.max_step = 0.01
        self.H = [qt.Qobj(np.zeros_like(self.H0))] + self.pulses  #We forget about the free Hamiltonian? Is this like in the interaction picture?
        self.evol = qt.mesolve(self.H, self.dmx, t, self.collapse, options=options_mesolve) 
        return self.evol
    
    def propagatorRWA(self,t):
        options_mesolve = qt.Options()
        options_mesolve.max_step = 0.01
        self.H = [qt.Qobj(np.zeros_like(self.H0))] + self.pulses
        self.propag = qt.propagator(self.H, t, self.collapse, options=options_mesolve)
        return self.propag
    
    def plotcohRWA(self, t):
        plt.figure()
        
        for i in range(int(2*self.s+1)):
           for j in range(i+1,int(2*self.s+1)):
              plt.plot(t,abs(self.get_evol(self.sigmas[i,j])))
            
        plt.show()

        plt.figure()
        
        for i in range(int(2*self.s+1)):
           for j in range(i+1,int(2*self.s+1)):
              plt.plot(t,(self.get_evol(self.sigmas[i,j])).real)
            
        plt.show()

        plt.figure()
        
        for i in range(int(2*self.s+1)):
           for j in range(i+1,int(2*self.s+1)):
              plt.plot(t,(self.get_evol(self.sigmas[i,j])).imag)
            
        plt.show()

    @staticmethod
    def ladder_system_adjacency_matrix(num_states : int) -> np.ndarray :
        """
        
        Create the adjacency matrix for a ladder system with the given number of states.

        Parameters:
        -----------
        num_states : int
            -- the number of states

        Returns:
        --------
        adj_matrix : np.ndarray
            -- the adjacency matrix

        """
        # Create the matrix
        adj_matrix = np.zeros((num_states, num_states))

        # Fill the matrix
        for i in range(num_states - 1):
            adj_matrix[i, i + 1] = 1
            adj_matrix[i + 1, i] = 1

        return adj_matrix

    @staticmethod
    def is_transition_graph_connected(adj_matrix : np.ndarray) -> bool:
        """

        Check if the transition graph is connected using Depth-First Search (DFS).
        
        DFS is a recursive algorithm that traverses the graph starting from a given vertex.
        It then recursively visits all the neighbors of that vertex, and so on.
        If all vertices are visited, then the graph is connected.
        Else, it is not connected.

        The function first checks that the given adjacency matrix is valid by calling is_valid_adjacency_matrix().

        Parameters:
        -----------
        adj_matrix : np.ndarray
            -- the adjacency matrix of the transition graph

        Returns:
        --------
        is_connected : bool
            -- True if the transition graph is connected, False otherwise

        """

        # Check that the adjacency matrix is valid
        if not Qudit.is_valid_adjacency_matrix(adj_matrix):
            return False

        # DFS algorithm
        def dfs(node):
            """
            Recursive DFS algorithm.

            Parameters:
            -----------
            node : int
                -- the current vertex

            Returns:
            --------
            None

            """
            visited[node] = True
            for neighbor, connected in enumerate(adj_matrix[node]):
                if connected and not visited[neighbor]:
                    dfs(neighbor)

        # Get the number of vertices
        num_vertices = len(adj_matrix)

        # Initialize the visited array
        visited = [False] * num_vertices

        # Start DFS from the first vertex
        dfs(0)

        # Check if all vertices are visited
        is_connected = all(visited)

        if not is_connected:
            raise ValueError('Invalid adjacency matrix: transition graph is not connected')

        return is_connected

    @staticmethod
    def is_valid_adjacency_matrix(adj_matrix : np.ndarray) -> bool:
        """
        
        Check if the given adjacency matrix is valid.

        The function first checks that all elements are 0 or 1.
        Then, it checks that the matrix is symmetric.

        Parameters:
        -----------
        adj_matrix : np.ndarray
            -- the adjacency matrix of the transition graph

        Returns:
        --------
        is_valid : bool
            -- True if the adjacency matrix is valid, False otherwise

        """
        # Check that all elements are 0 or 1
        if not all(0 <= value <= 1 for row in adj_matrix for value in row):
            raise ValueError('Invalid adjacency matrix: all elements must be 0 or 1')

        # Check that the matrix is symmetric
        if not np.allclose(adj_matrix, adj_matrix.T):
            raise ValueError('Invalid adjacency matrix: must be symmetric')

        return True
        
def pulse(ti, dur, form=np.sin, step=None, offset=False):
    if not step: step = lambda t: np.heaviside(t-ti,0)-np.heaviside(t-ti-dur,0)
    offs = ti*offset
    def ft(t,args=0):
        return form(t-ti)*step(t-offs)
    return ft

def constpulse(ti, dur, Om=1, step=None, offset=False):
    def form(t):
        return Om
    return pulse(ti, dur, form, step, offset)

def trigpulse(ti, dur, w, Om=1, phi=0, step=None, offset=False):
    def form(t):
        return Om*np.cos(w*t+phi)
    return pulse(ti, dur, form, step, offset)

def exppulse(ti, dur, w, Om=1, phi=0, step=None, offset=False):
    def form(t):
        return Om*np.exp(w*1.0j*t+phi)
    return pulse(ti, dur, form, step, offset)