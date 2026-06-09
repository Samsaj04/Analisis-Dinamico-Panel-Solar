import numpy as np
from scipy.optimize import root_scalar
import matplotlib.pyplot as plt
from functools import lru_cache
from cycler import cycler

plt.style.use('seaborn-v0_8-whitegrid')

plt.rcParams.update({
    "font.size": 14,
    "axes.labelsize": 15,
    "axes.titlesize": 16,
    "legend.fontsize": 12
})

plt.rcParams['axes.prop_cycle'] = cycler(color=[
    "#4C72B0",
    "#DD8452",
    "#55A868",
    "#C44E52",
    "#8172B3",
    "#937860",
    "#DA8BC3",
    "#8C8C8C" 
])

class Material:
    def __init__(self, name, E, rho, zeta):
        self.name = name
        self.E = E          #Modulo de Young [Pa]
        self.rho = rho      #Densidad Volumetrica [kg/m^3]
        self.zeta = zeta    #Tasa de Amortiguamiento
        
class Conditions:
    def __init__(self, mat, mat_list, L, b, w, ms=None, t=None):
        self.mat = mat  #Material de entrada
        self.mat_list = mat_list #Lista de todos los materiales
        
        self.L = L  #Largo [m]
        self.b = b  #Ancho [m]
        self.w = w  #Alto  [m]
        
        self.ms = ms #Masa [kg] (sobreescribir en caso de no tener densidad uniforme)
        self.t = t  #Espesor [m] (>0 si se usa viga hueca)
        
    #Decorador property para convertir atributos en propiedades dinámicas
    @property
    def A(self): #Area Transversal [m^2]
        if self.t is None:
            return self.b*self.w
        return self.b*self.w - (self.b-2*self.t)*(self.w-2*self.t)

    @property
    def m(self): #Masa [kg]
        if self.ms is None:
            return self.A*self.L*self.new_rho()
        return self.ms

    def new_rho(self):
        if self.ms is None:
            return self.mat.rho
        return self.ms/(self.A * self.L)
    
    @property
    def I(self): #Inercia de Area [m^4]
        if self.t is None:
            return 1/12 * self.b * self.w**3
        return (self.b * self.w**3 - (self.b-2*self.t)*(self.w-2*self.t)**3) / 12
        
    # Término Bn*L para cualquier entero "n"
    @lru_cache(maxsize=None) #Decorador de memoria cache para optimizar la funcion
    def Bn_L(self, n):
        if n < 1:
            return np.nan
        F = lambda x: np.cos(x)*np.cosh(x) + 1
        arr = np.linspace(0, 3.5*n, 6000)
        inte = []
        for i in range(len(arr)-1):
            if F(arr[i]) * F(arr[i+1]) < 0:
                inte.append((arr[i], arr[i+1]))
        a, b = inte[n-1]
        return root_scalar(F, bracket=[a, b]).root
    
    # Función de modo Wn(x), dependiendo del modo natural del panel
    def Wn(self, N, x):
        Bn = self.Bn_L(N)/self.L
        A = np.cos(Bn*x) - np.cosh(Bn*x)
        B = (np.cos(Bn*self.L) + np.cosh(Bn*self.L))/(np.sin(Bn*self.L) + np.sinh(Bn*self.L))
        C = np.sin(Bn*x) - np.sinh(Bn*x)
        return A - B*C
    
    def bending_stress(self, N, x):
        Bn = self.Bn_L(N) / self.L
        A = np.cos(Bn*x) + np.cosh(Bn*x)
        ft = (np.cos(Bn*self.L) + np.cosh(Bn*self.L)) / (np.sin(Bn*self.L) + np.sinh(Bn*self.L))
        C = np.sin(Bn*x) + np.sinh(Bn*x)
        return self.mat.E * self.w/2 * Bn**2 * (A - ft*C)
        
    # Función de la frecuencia angular del panel oscilando en modo natural
    def omega(self, N):
        return self.Bn_L(N)**2 * np.sqrt(self.mat.E * self.I / (self.new_rho() * self.A * self.L**4))
    
    # Frecuencia del panel oscilando en modo natural
    def Fn(self, N):
        return self.omega(N)/(2*np.pi)
    
    # Relación de frecuencias de vibracion externa entre vibracion natural
    def frec_ratio(self, w, N):
        return w/self.omega(N)
    
    # Factor de amplificacion de amplitud de oscilacion segun el amortiguamiento y relacion de recuencias
    def FM(self, R, zeta):
        return 1 / np.sqrt((1-R**2)**2 + (2*zeta*R)**2)
    
    #============PLOTS==================
    
    #Perfil de deflexion de la viga con respecto a su longitud (Deflexión normalizada)
    def plot_Wn(self, rang, save=False):
        X = np.linspace(0.0001, self.L, 500)
        fig, ax = plt.subplots(figsize=(8,5))
        for i in range(rang[0], rang[1]+1):
            Xn = self.Wn(i, X)
            plt.plot(X, Xn/np.max(np.abs(Xn)), label=f"n = {i}, $B_nL$ = {self.Bn_L(i)}")
        plt.xlabel("Longitud de la Viga [m]")
        plt.ylabel("Amplitud Normalizada")
        #plt.title(f"Desplazamiento Transversal Normalizado vs Longitud de Viga en Voladizo")
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend()
        if save:
            plt.savefig("fig1_formas_modales.pdf", dpi=300)
        plt.show()

    #Gráfica del Factor de Amplificación con respecto a la tasa de frecuencias
    def plot_FM(self, save=False):
        R = np.linspace(0.5, 1.5, 200000)
        fig, ax = plt.subplots(figsize=(8,5))
        for mat in self.mat_list:
            Mn = self.FM(R, mat.zeta)
            plt.plot(R, Mn, label=f"{mat.name}, $\\zeta$ = {mat.zeta}")
            
        plt.axvline(x=1, linestyle=":", linewidth=2, color='purple', label="Resonancia Pura")
        plt.axvline(x=0.8, linestyle=":", linewidth=2, color='red', alpha=0.6)
        plt.axvline(x=1.2, linestyle=":", linewidth=2, color='red', alpha=0.6)
        
        #plt.xticks(np.arange(0, 2+0.1, 0.2))
        #plt.yscale('log')
        plt.xlim(0.995, 1.005)
        plt.xlabel("Relación de Frecuencias [$\\frac{w}{w_n}$]")
        plt.ylabel("Factor de Magnificación [M]")
        #plt.title("Tasa de Frecuencias vs Factor de Magnificación")
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend()
        if save:
            plt.savefig("fig2_factor_magnificacion.pdf", dpi=300)
        plt.show()

    #Gráfica de Frecuencia Natural contra Alto del Panel
    def plot_Fn_vs_w(self, f, rang, save=False):
        H = np.linspace(0.001, 0.03, 600)
        w0 = self.w #Guardo en memoria mi espesor inicial
        fig, ax = plt.subplots(figsize=(8,5))
        try:
            for i in range(rang[0], rang[1]+1):
                self.w = H # Asignación temporal de rango de espesores a self.w
                plt.plot(self.w * 1000, self.Fn(i), label=f"n = {i}")
        finally:
            self.w = w0 #Restauro espesor a su condicion inicial
        plt.axhspan(f[0], f[1], color="red", alpha=0.2, label=f"Rango de Frecuencias ({f[0]}, {f[1]})")
        plt.xticks(np.arange(1, 32, 2))
        plt.xlabel("Espesor del Panel [mm]")
        plt.ylabel("Frecuencia Natural $F_n$ [Hz]")
        #plt.title(f"Espesor ($h$) vs Frecuencia Natural - ({self.mat.name})")
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend()
        if save:
            plt.savefig("fig3_frecuencia_vs_espesor.pdf", dpi=300)
        plt.show()

    #Grafica Frecuencia de un modo N variando el espesor para cada material
    def plot_Fn_vs_h_mats(self, f, rango, save=False):
        H = np.linspace(0.001, 0.03, 600)
        fig, ax = plt.subplots(figsize=(8,5))
        w0 = self.w
        mat0 = self.mat
        line = ['-', '--', ':', '-.']
        try:
            self.w = H
            for l, N in enumerate(range(rango[0], rango[1]+1)):
                for m in self.mat_list:
                    self.mat = m
                    plt.plot(H * 1000, self.Fn(N), linestyle=line[l%len(line)], label=f"{m.name} - n = {N}")
        finally:
            self.w = w0
            self.mat = mat0
        plt.axhspan(f[0], f[1], color="red", alpha=0.2, label=f"Rango de Frecuencias ({f[0]}, {f[1]})")
        plt.xticks(np.arange(0, 30, 2)) 
        plt.xlabel("Espesor del Panel [mm]")
        plt.ylabel(f"Frecuencia Natural $F_n$ [Hz] (Modo {N})")
        #plt.title(f"Espesor vs Frecuencia Natural entre Materiales")
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend()
        if save:
            plt.savefig("fig4_materiales_vs_espesor.pdf", dpi=300)
        plt.show()
        
    #Gráfica del Esfuerzo Flector Relativo en la Longitud de la Viga
    def plot_BStress(self, rang, save=False):
        fig, ax = plt.subplots(figsize=(8,5))
        X = np.linspace(0.0001, self.L, 500)
        for i in range(rang[0], rang[1]+1):
            sigma = self.bending_stress(i, X)
            sigma /= np.max(np.abs(sigma))
            plt.plot(X, sigma, label=f"n = {i}")
        plt.xlabel("Longitud del Panel [m]")
        plt.ylabel("Esfuerzo Flector")
        #plt.title(f"Esfuerzo Flector Normalizado vs Longitud de Panel - ({self.mat.name})")
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend()
        if save:
            plt.savefig("fig5_esfuerzo_flector.pdf", dpi=300)
        plt.show()
        
    def plots_log(self, f, rang, save=False):
        self.plot_Wn(rang, save)
        self.plot_FM(save)
        self.plot_Fn_vs_w(f, rang, save)
        self.plot_Fn_vs_h_mats(f, rang, save)
        self.plot_BStress(rang, save)

#====================================================================
#====================================================================
def main():
    alum = Material(
        name="Al6061-T6",
        E=68.9e09,  
        rho=2700,
        zeta=0.000625)

    tit = Material(
        name="Titanium Alloy",
        E=114e09,  
        rho=4430,
        zeta=0.00015)

    CFRP = Material(
        name="Carbon Fiber",
        E=150e09,  
        rho=1600,
        zeta=0.0009)

    AL_alloy = Material(
        name="Aluminum Alloy",
        E=140e09,  
        rho=1389.58,
        zeta=0.000625)

    S2 = Conditions(
        mat=AL_alloy,
        mat_list=[alum, tit, CFRP],
        L=1.1,
        b=20.3/1000,
        w=20.3/1000,
        t=1.27/1000,
        ms=1.68)

    rango_frec = [20, 100]
    ran = [1, 3]
    
    S2.plots_log(rango_frec, ran, save=True)
    
if __name__ == "__main__":
    main()
