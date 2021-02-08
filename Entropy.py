from packman import molecule

from scipy import spatial
from scipy.spatial import Voronoi
from scipy.spatial import ConvexHull

from multiprocessing.dummy import Pool as ThreadPool
from functools import partial

from numpy import pi, cos, sin, arccos, arange
import numpy



def GetSpherePoints(point,tree,radius=1.4,num_pts=30):
    indices = arange(0, num_pts, dtype=float) + 0.5
    phi = arccos(1 - 2*indices/num_pts)
    theta = pi * (1 + 5**0.5) * indices
    x, y, z = cos(theta) * sin(phi), sin(theta) * sin(phi), cos(phi)
    x,y,z=radius*x,radius*y,radius*z
    x=x+point[0]
    y=y+point[1]
    z=z+point[2]
    data = zip(x.ravel(), y.ravel(), z.ravel())
    selected=[]
    for _ in data:
        if(tree.query(_,k=2)[0][1]>radius):
            selected.append(_)
    return selected


def GetSurfacePoints(atoms,probe_size=1.4):
    coordinates=[i.get_location() for i in atoms]
    tree = spatial.KDTree(coordinates) 
    pool = ThreadPool(10)
    surface=pool.map(partial(GetSpherePoints,radius=probe_size, tree=tree), coordinates)
    surface=[item for sublist in surface for item in sublist]
    return surface


def GetVoronoiInfo(atoms,probe_size=1.4):
    '''
    Margin should never be less than 1
    '''
    atoms=[i for i in atoms]

    Surface=GetSurfacePoints(atoms,probe_size=probe_size)
    points=numpy.concatenate((Surface,[i.get_location() for i in atoms]))
    voronoi=Voronoi(points)

    AvailableVolume={j:[] for j in set([i.get_parent() for i in atoms])}
    for numi, i in enumerate(atoms):
        #NOTE: Check if -1 is present, we put border to solve the issue but check it anyways
        atom_cell=[]
        for j in voronoi.regions[voronoi.point_region[numi+len(Surface)]]:
            atom_cell.append(voronoi.vertices[j])
        AvailableVolume[i.get_parent()].extend(atom_cell)

    #Delete this part later
    '''
    InteractionMatrix=numpy.zeros((len(AvailableVolume),len(AvailableVolume)),dtype=int)
    for numi,i in enumerate(AvailableVolume):
        for numj,j in enumerate(AvailableVolume):
            if(numi!=numj):
                InteractionMatrix[numi][numj]=len(numpy.intersect1d(AvailableVolume[i],AvailableVolume[j]))
    
    for i in range(0,len(InteractionMatrix)):InteractionMatrix[i][i]=numpy.sum(InteractionMatrix[i])
    numpy.savetxt('delete.txt',InteractionMatrix,fmt='%i')
    '''
    
    #Each value= Total volume of voronoi cells of the atoms of the residue / Total volume of the residues (Both convex hull volumes)
    PackingFraction={}
    for i in AvailableVolume:
        #print(i.get_name(),float(ConvexHull([j.get_location() for j in i.get_atoms()]).volume) / ConvexHull(AvailableVolume[i]).volume)
        #NOTE: Find better alternative to the ConvexHull !!!!!!! VERY IMPORTANT CHANGE; MAKE SURE YOU READ THIS
        PackingFraction[i]=float(ConvexHull([j.get_location() for j in i.get_atoms()]).volume) / ConvexHull(AvailableVolume[i]).volume
        #NOTE: Following lines are to find better alternative to the convex hull for occupied volume
        #radius=1
        #PackingFraction[i]=float(numpy.sum([(4/3)*numpy.pi*(radius**3) for j in i.get_atoms()])) / ConvexHull(AvailableVolume[i]).volume

    #p log p
    Entropy={i:-PackingFraction[i]*numpy.log(PackingFraction[i]) for i in PackingFraction}
    return Entropy




def Single(filename='1exr.pdb',chain=None):
    '''
    '''
    try:
        molecule.download_structure(filename[:4])
    except:
        None
    mol = molecule.load_structure(filename[:4]+'.cif')
    
    if(chain!=None):
        all_atoms=[j for i in mol[0][chain].get_residues() for j in i.get_atoms()]
    else:
        all_atoms=[j for i in mol[0].get_residues() for j in i.get_atoms()]
    Voronoi_info=GetVoronoiInfo(all_atoms,probe_size=1.4)

    Voronoi_info_keys=[i for i in Voronoi_info]
    Voronoi_info_keys_id=[i.get_id() for i in Voronoi_info_keys]
    Voronoi_info_keys=[x for _,x in sorted(zip(Voronoi_info_keys_id,Voronoi_info_keys))]
    
    #for i in Voronoi_info_keys:
    #    print(i.get_name()+'-'+str(i.get_id()),Voronoi_info[i])
    
    vor=[Voronoi_info[i] for i in Voronoi_info_keys]
    #Total packing entropy
    #print(filename,chain,numpy.sum(vor))
    return numpy.sum(vor)

def main():
    #Single Example
    print( Single('1exr', 'A') )

    #Multiple
    '''
    for i in open('List.txt').readlines():
        arg=i.strip().split()
        try:
            Single(arg[0],arg[1])
        except:
            None
    return True
    '''

if(__name__=='__main__'):
    main()