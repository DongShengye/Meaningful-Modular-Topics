import numpy as np
import os


class Ply(object):
    """Class to represent a ply in memory, read plys, and write plys.
    """

    def __init__(self, ply_path=None, triangles=None, points=None, normals=None, colors=None):
        """Initialize the in memory ply representation.

        Args:
            ply_path (str, optional): Path to .ply file to read (note only
                supports text mode, not binary mode). Defaults to None.
            triangles (numpy.array [k, 3], optional): each row is a list of point indices used to
                render triangles. Defaults to None.
            points (numpy.array [n, 3], optional): each row represents a 3D point. Defaults to None.
            normals (numpy.array [n, 3], optional): each row represents the normal vector for the
                corresponding 3D point. Defaults to None.
            colors (numpy.array [n, 3], optional): each row represents the color of the
                corresponding 3D point. Defaults to None.
        """
        

        super().__init__()

        # TODO: If ply path is None, load in triangles, point, normals, colors.
        #       else load ply from file. If ply_path is specified AND other inputs
        #       are specified as well, ignore other inputs.
        # TODO: If normals are not None make sure that there are equal number of points and normals.
        # TODO: If colors are not None make sure that there are equal number of colors and normals.
        
        
        self.file2write = ''#Heading file

        self.ply_path = ply_path
        self.points = points
        self.triangles = triangles
        self.normals = normals
        self.colors = colors

        if self.ply_path == None or os.path.exists(self.ply_path) == False:

            if points is None:
                self.pointcount = 0
            else:
                self.pointcount = np.shape(points)[0]

            if triangles is None:
                self.facecount = 0
            else:
                self.facecount = np.shape(triangles)[0] 

            if normals is None:
                self.normalcount = 0
            else:
                self.normalcount = np.shape(normals)[0] 
                if self.normalcount != self.pointcount:
                    
                    raise Exception("issue warning-self.normals != self.pointcount")

            if colors is None:
                self.colorcount = 0
            else:
                self.colorcount = np.shape(colors)[0]           
                if self.normalcount != self.colorcount:
                            
                    raise Exception("issue warning-self.normals != self.colors")
            #print('color',self.colors)
            #print(self.normals)

            pass
        
    def write(self,ply_path):
        """Write mesh, point cloud, or oriented point cloud to ply file.

        Args:
            ply_path (str): Output ply path.
        """
        # TODO: Write header depending on existance of normals, colors, and triangles.


        def aiwriter1(a,b,c): # x y z nx ny nz red green blue

            text = ''
            shapea = np.shape(a)
            shapeb = np.shape(b)
            shapec = np.shape(c)
            # print(shapea)
            # print(shapeb)
            # print(shapec)
            for i in range(shapea[0]):
                for j in range(shapea[1]):
                        text+=f'{a[i][j]} '
                
                if shapeb:
                    for j in range(shapeb[1]):
                            text+=f'{b[i][j]} '
                if shapec:
                    for j in range(shapec[1]):
                        # if j != shapec[1]-1:
                        text+=f'{c[i][j]} '
                text+='\n'
                    
            return text

        def aiwriter2(d): # vertex

            
            shaped = np.shape(d)
            
            print("shaped",shaped)
            text = ''
            for i in range(shaped[0]):
                text+=f'{shaped[1]} '
                for j in range(shaped[1]):

                    text+=f'{d[i][j]} '
                text+='\n'
                        
            return text

        self.file2write+=f"ply\nformat ascii 1.0\nelement vertex {self.pointcount}\nproperty float x\nproperty float y\nproperty float z\nproperty float nx\
        \nproperty float ny\nproperty float nz\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nelement face {self.facecount}\nproperty list uchar int vertex_index\nend_header\n"

        self.file2write+=aiwriter1(self.points,self.normals,self.colors)
        shaped = np.shape(self.triangles) #possible error
        if shaped:
            self.file2write+=aiwriter2(self.triangles)
            print(self.file2write)

        with open(ply_path, 'w') as fp:
            fp.write(self.file2write)

        pass

    def read(self, ply_path):
        """Read a ply into memory.

        Args:
            ply_path (str): ply to read in.
        """
        # TODO: Read in ply.

        with open(ply_path, "r") as file:
  
            self.file2write  = file.read()
        print(self.file2write)
        pass

pp = Ply(ply_path=None, triangles=np.array([[2,1,0]]), points=np.array([[0,0,1],[0,1,0],[1,0,0]]), normals=np.array([[1,0,0],[1,0,0],[1,0,0]]), colors=np.array([[0,0,155],[0,0,155],[0,0,155]]))
#pp = Ply(ply_path=None, triangles=None, points=np.array([[0,0,1],[0,0,1],[0,1,0],[1,0,0]]), normals=np.array([[1,0,0],[1,0,0],[1,0,0],[1,0,0]]), colors=np.array([[0,0,155],[0,0,155],[0,0,155],[0,0,155]]))
#pp.read("C:/Users/bests/OneDrive/Desktop/hw1/hw1/data/triangle_sample.ply")
#pp.write("C:/Users/bests/OneDrive/Desktop/hw1/hw1/data/test_point.ply")
