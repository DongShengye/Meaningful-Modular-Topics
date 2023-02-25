
from skimage import measure
from transforms import *


class TSDFVolume:
    """Volumetric TSDF Fusion of RGB-D Images.
    """

    def __init__(self, volume_bounds, voxel_size):
        """Initialize tsdf volume instance variables.

        Args:
            volume_bounds (numpy.array [3, 2]): rows index [x, y, z] and cols index [min_bound, max_bound].
                Note: units are in meters.
            voxel_size (float): The side length of each voxel in meters.

        Raises:
            ValueError: If volume bounds are not the correct shape.
            ValueError: If voxel size is not positive.
        """
        volume_bounds = np.asarray(volume_bounds)
        if volume_bounds.shape != (3, 2):
            raise ValueError('volume_bounds should be of shape (3, 2).')

        if voxel_size <= 0.0:
            raise ValueError('voxel size must be positive.')

        # Define voxel volume parameters
        self._volume_bounds = volume_bounds
        self._voxel_size = float(voxel_size)
        self._truncation_margin = 2 * self._voxel_size  # truncation on SDF (max alowable distance away from a surface)

        # Adjust volume bounds and ensure C-order contiguous
        # and calculate voxel bounds taking the voxel size into consideration
        self._voxel_bounds = np.ceil(
            (self._volume_bounds[:, 1] - self._volume_bounds[:, 0]) / self._voxel_size
        ).copy(order='C').astype(int)
        self._volume_bounds[:, 1] = self._volume_bounds[:, 0] + self._voxel_bounds * self._voxel_size

        # volume min bound is the origin of the volume in world coordinates
        self._volume_origin = self._volume_bounds[:, 0].copy(order='C').astype(np.float32)

        print('Voxel volume size: {} x {} x {} - # voxels: {:,}'.format(
            self._voxel_bounds[0],
            self._voxel_bounds[1],
            self._voxel_bounds[2],
            self._voxel_bounds[0] * self._voxel_bounds[1] * self._voxel_bounds[2]))

        # Initialize pointers to voxel volume in memory
        self._tsdf_volume = np.ones(self._voxel_bounds).astype(np.float32)

        # for computing the cumulative moving average of observations per voxel
        self._weight_volume = np.zeros(self._voxel_bounds).astype(np.float32)
        color_bounds = np.append(self._voxel_bounds, 3)
        self._color_volume = np.zeros(color_bounds).astype(np.float32)  # rgb order

        # Get voxel grid coordinates
        xv, yv, zv = np.meshgrid(
            range(self._voxel_bounds[0]),
            range(self._voxel_bounds[1]),
            range(self._voxel_bounds[2]),
            indexing='ij')
        self._voxel_coords = np.concatenate([
            xv.reshape(1, -1),
            yv.reshape(1, -1),
            zv.reshape(1, -1)], axis=0).astype(int).T

    def get_volume(self):
        """Get the tsdf and color volumes.

        Returns:
            numpy.array [l, w, h]: l, w, h are the dimensions of the voxel grid in voxel space.
                Each entry contains the integrated tsdf value.
            numpy.array [l, w, h, 3]: l, w, h are the dimensions of the voxel grid in voxel space.
                3 is the channel number in the order r, g, then b.
        """

        return self._tsdf_volume, self._color_volume

    def get_mesh(self):
        """ Run marching cubes over the constructed tsdf volume to get a mesh representation.

        Returns:
            numpy.array [n, 3]: each row represents a 3D point.
            numpy.array [k, 3]: each row is a list of point indices used to render triangles.
            numpy.array [n, 3]: each row represents the normal vector for the corresponding 3D point.
            numpy.array [n, 3]: each row represents the color of the corresponding 3D point.
        """
        tsdf_volume, color_vol = self.get_volume()

        # Marching cubes
        voxel_points, triangles, normals, _ = measure.marching_cubes(tsdf_volume, level=0, method='lewiner')
        points_ind = np.round(voxel_points).astype(int)
        points = self.voxel_to_world(self._volume_origin, voxel_points, self._voxel_size)

        # Get vertex colors.
        rgb_vals = color_vol[points_ind[:, 0], points_ind[:, 1], points_ind[:, 2]]
        colors_r = rgb_vals[:, 0]
        colors_g = rgb_vals[:, 1]
        colors_b = rgb_vals[:, 2]
        colors = np.floor(np.asarray([colors_r, colors_g, colors_b])).T
        colors = colors.astype(np.uint8)

        return points, triangles, normals, colors

    """
    *******************************************************************************
    ****************************** ASSIGNMENT BEGINS ******************************
    *******************************************************************************
    """

    @staticmethod
    @njit(parallel=True)
    def voxel_to_world(volume_origin, voxel_coords, voxel_size):
        """ Convert from voxel coordinates to world coordinates
            (in effect scaling voxel_coords by voxel_size).

        Args:
            volume_origin (numpy.array [3, ]): The origin of the voxel
                grid in world coordinate space.
            voxel_coords (numpy.array [n, 3]): Each row gives the 3D coordinates of a voxel.
            voxel_size (float): The side length of each voxel in meters.

        Returns:
            numpy.array [n, 3]: World coordinate representation of each of the n 3D points.
        """
        volume_origin = volume_origin.astype(np.float32)
        print("volume_origin",volume_origin)
        voxel_coords = voxel_coords.astype(np.float32)
        world_points = np.empty_like(voxel_coords, dtype=np.float32)

        # NOTE: prange is used instead of range(...) to take advantage of parallelism.
        for i in prange(voxel_coords.shape[0]):
            # TODO:
            world_points[i] = voxel_coords[i]*voxel_size+volume_origin

            pass
        return world_points

    @staticmethod
    @njit(parallel=True)
    def get_new_tsdf_and_weights(tsdf_old, margin_distance, w_old, observation_weight):
        """[summary]

        Args:
            tsdf_old (numpy.array [v, ]): v is equal to the number of voxels to be
                integrated at this timestamp. Old tsdf values that need to be
                updated based on the current observation.
            margin_distance (numpy.array [v, ]): The tsdf values of the current observation.
                It should be of type numpy.array [v, ], where v is the number
                of valid voxels.
            w_old (numpy.array [v, ]): old weight values.
            observation_weight (float): Weight to give each new observation.

        Returns:
            numpy.array [v, ]: new tsdf values for entries in tsdf_old
            numpy.array [v, ]: new weights to be used in the future.
        """
        tsdf_new = np.empty_like(tsdf_old, dtype=np.float32)
        w_new = np.empty_like(w_old, dtype=np.float32)

        for i in prange(len(tsdf_old)):
            
        # TODO: 
            
            w_new[i] = w_old[i]+observation_weight
            tsdf_new[i] = (w_old[i]*tsdf_old[i]+observation_weight*margin_distance[i])/w_new[i]
            
        return tsdf_new, w_new

    def get_valid_points(self, depth_image, voxel_u, voxel_v, voxel_z):
        """ Compute a boolean array for indexing the voxel volume and other variables.
        Note that every time the method integrate(...) is called, not every voxel in
        the volume will be updated. This method returns a boolean matrix called
        valid_points with dimension (n, ), where n = # of voxels. Index i of
        valid_points will be true if this voxel will be updated, false if the voxel
        needs not to be updated.

        The criteria for checking if a voxel is valid or not is shown below.

        Args:
            depth_image (numpy.array [h, w]): A z depth image.          #Something wrong here
            voxel_u (numpy.array [v, ]): Voxel coordinate projected into image coordinate, axis is u
            voxel_v (numpy.array [v, ]): Voxel coordinate projected into image coordinate, axis is v
            voxel_z (numpy.array [v, ]): Voxel coordinate projected into camera coordinate axis z
        Returns:
            valid_points numpy.array [v, ]: A boolean matrix that will be
            used to index into the voxel grid. Note the dimension of this
            variable.
        """

        image_height, image_width = depth_image.shape
        print("image_height, image_width",image_height, image_width)
        # TODO 1:
        #  Eliminate pixels not in the image bounds or that are behind the image plane
        truecounter1 = 0
        b1 = []
        for i in range(len(voxel_u)):
            #print(voxel_u[i])
            if voxel_u[i]>=image_width or voxel_u[i]<=0 or voxel_v[i]>=image_height or voxel_v[i]<=0 or voxel_z[i]<=0:
                b1.append(False)

            else:
                b1.append(True)
                truecounter1+=1
                
        b1 = np.array(b1)
        # TODO 2.1:
        #  Get depths for valid coordinates u, v from the depth image. Zero elsewhere.
        depth_list = []
        b2 = []
        truecounter2 = 0
        for i in range(len(voxel_u)):

            try:
                #print(int(voxel_v[i]))
                depth_list.append(depth_image[int(voxel_v[i])][int(voxel_u[i])]) #be careful here. Maybe depth image only int indices
                
                b2.append(True)
                truecounter2+=1
            except:
                depth_list.append(0)
                b2.append(False)
            #print(len(b2))
                
        b2 = np.array(b2)
        print("truecounter1",truecounter1,"truecounter2",truecounter2)
        # TODO 2.2:
        #  Calculate depth differences
        #depth_diff = np.array(depth_list)-voxel_z[b1 & b2]

        # TODO 2.3:
        #  Filter out zero depth values 
        
        return b1 & b2




    # @staticmethod
    # @njit(parallel=True)
    def get_new_colors_with_weights(color_old, color_new, w_old, w_new, observation_weight):
        """ Compute the new RGB values for the color volume given the current values
        in the color volume, the RGB image pixels, and the old and new weights.

        Args:
            color_old (numpy.array [n, 3]): Old colors from self._color_volume in RGB.
            color_new (numpy.array [n, 3]): Newly observed colors from the image in RGB
            w_old (numpy.array [n, ]): Old weights from the self._tsdf_volume
            w_new (numpy.array [n, ]): New weights from calling get_new_tsdf_and_weights
            observation_weight (float, optional):  The weight to assign for the current
                observation. Defaults to 1.
        Returns:
            valid_points numpy.array [n, 3]: The newly computed colors in RGB. Note that
            the input color and output color should have the same dimensions.
        """

        # TODO: Compute the new R, G, and B value by summing the old color
        #  value weighted by the old weight, and the new color weighted by
        #  observation weight. Finally normalize the sum by the new weight.
        color_filtered = np.empty_like(color_new)
        w_old = np.vstack((np.vstack((w_old,w_old)),w_old)).T
        print("w_old",np.shape(w_old))
        w_new = np.vstack((np.vstack((w_new,w_new)),w_new)).T
        print("w_new",np.shape(w_new))
        print("color_old*w_old",np.shape(color_old*w_old))
        print("color_new*observation_weight",np.shape(color_new*observation_weight))
        print("w_new",w_new)
        color_filtered = (color_old*w_old+color_new*observation_weight)/w_new
        
        
            
        return color_filtered    

        pass

    def integrate(self, color_image, depth_image, camera_intrinsics, camera_pose, observation_weight=1.):
        """Integrate an RGB-D observation into the TSDF volume, by updating the weight volume,
            tsdf volume, and color volume.

        Args:
            color_image (numpy.array [h, w, 3]): An rgb image.
            depth_image (numpy.array [h, w]): A z depth image.
            camera_intrinsics (numpy.array [3, 3]): given as [[fu, 0, u0], [0, fv, v0], [0, 0, 1]]
            camera_pose (numpy.array [4, 4]): SE3 transform representing pose (camera to world)
            observation_weight (float, optional):  The weight to assign for the current
                observation. Defaults to 1.
        """
        color_image = color_image.astype(np.float32)

        # TODO: 1. Project the voxel grid coordinates valid pixelsto the world
        #  space by calling `voxel_to_world`. Then, transform the points
        #  in world coordinate to camera coordinates, which are in (u, v).
        #  You might want to save the voxel z coordinate for later use.

        #world_points: World coordinate representation of each of the n 3D points
        world_points = self.voxel_to_world(self._volume_origin, self._voxel_coords, self._voxel_size) 
        print("world_points",np.shape(world_points))
        print()
        print("self._volume_origin",self._volume_origin)
        print("self._voxel_coords",self._voxel_coords)
        print("self._voxel_size",self._voxel_size)
        print("world_points",np.shape(world_points))
        #voxel_z: z-coordinates of voxels in the world frame-prepare for future use
        
        # TODO: 1.5. 
        #world ->camera
        voxel_camera = np.linalg.inv(camera_pose)@np.column_stack((world_points,np.ones(np.shape(world_points)[0]))).T
        
        voxel_camera = voxel_camera[0:3,:]
        voxel_z_image = voxel_camera[2,:].T
        print("voxel_z_image",np.shape(voxel_z_image))
        voxel_camera = voxel_camera.T
          #What is voxel_z???
        print("voxel_camera",np.shape(voxel_camera))
        #camera ->image
        def camera_to_image(intrinsics, camera_points):
            """Project points in camera space to the image plane.

            Args:
                intrinsics (numpy.array [3, 3]): Pinhole intrinsics.
                camera_points (numpy.array [n, 3]): n 3D points (x, y, z) in camera coordinates.

            Raises:
                ValueError: If intrinsics are not the correct shape.
                ValueError: If camera points are not the correct shape.

            Returns:
                numpy.array [n, 2]: n 2D projections of the input points on the image plane.
            """
            if intrinsics.shape != (3, 3):
                raise ValueError('Invalid input intrinsics')
            if len(camera_points.shape) != 2 or camera_points.shape[1] != 3:
                raise ValueError('Invalid camera point')

            u0 = intrinsics[0, 2]
            v0 = intrinsics[1, 2]
            fu = intrinsics[0, 0]
            fv = intrinsics[1, 1]

            # find u, v int coords
            image_coordinates = np.empty((camera_points.shape[0], 2), dtype=np.int64)
            for i in prange(camera_points.shape[0]):
                try:
                    image_coordinates[i, 0] = int(np.round((camera_points[i, 0] * fu / camera_points[i, 2]) + u0))
                    image_coordinates[i, 1] = int(np.round((camera_points[i, 1] * fv / camera_points[i, 2]) + v0))
                except:
                    image_coordinates[i, 0] = 100000000000
                    image_coordinates[i, 1] = 100000000000
            return image_coordinates
        

        voxel_image_2d = camera_to_image(camera_intrinsics,voxel_camera)
        print("voxel_image_2d",voxel_image_2d)




        
        voxel_u_image = voxel_image_2d[:, 0]
        voxel_v_image = voxel_image_2d[:, 1]


        print("voxel_u_image",np.shape(voxel_u_image))
        print("voxel_u_image",voxel_u_image)
        print("voxel_v_image",np.shape(voxel_v_image))
        print("voxel_v_image",voxel_v_image)
        print("voxel_z_image",np.shape(voxel_z_image))
        print("voxel_z_image",voxel_z_image)

        # TODO: 2.
        #  Get all of the valid points in the voxel grid by implementing
        #  the helper get_valid_points. Be sure to pass in the correct parameters.



        bool_valid = self.get_valid_points(depth_image, voxel_u_image, voxel_v_image, voxel_z_image)
        counter = 0
        for i in range(len(bool_valid)):
            if bool_valid[i] == True:
                counter+=1
        print("True number",counter)
        print("bool_valid",np.shape(bool_valid))
        # TODO: 3.
        #  With the valid_points array as your indexing array, index into
        #  the self._voxel_coords variable to get the valid voxel x, y, and z.
        valid_voxel = self._voxel_coords[bool_valid]
        print("valid_voxel",np.shape(valid_voxel))
        # TODO: 4. With the valid_points array as your indexing array,
        #  get the valid pixels. Use those valid pixels to index into
        #  the depth_image, and find the valid margin distance.
        valid_pixel = np.vstack((np.vstack((voxel_u_image[bool_valid],voxel_v_image[bool_valid])),voxel_z_image[bool_valid])).T
        
        print("valid_pixel",np.shape(valid_pixel)) 
        margin_valid = []
        valid_pixel_2d = np.vstack((voxel_u_image[bool_valid],voxel_v_image[bool_valid])).T.astype(np.int)
        print("valid_pixel_2d",np.shape(valid_pixel_2d))
        print("depth map",np.shape(depth_image))
        
        
        margin_valid = depth_image[valid_pixel_2d[:,1],valid_pixel_2d[:,0]]-voxel_z_image[bool_valid]
        margin_valid[margin_valid>2.5] = 2.5
        max_value = margin_valid.max()
        margin_valid = margin_valid/max_value
        print("care for margin_valid,care for margin_valid,care for margin_valid,care for margin_valid,care for margin_valid,care for margin_valid,care for margin_valid,care for margin_valid,care for margin_valid,care for margin_valid,")
        print(margin_valid)
        print("margin_valid",np.shape(margin_valid))


        # TODO: 5.
        #  Compute the new weight volume and tsdf volume by calling
        #  `get_new_tsdf_and_weights`. Then update the weight volume
        #  and tsdf volume.

        """[summary]

        Args:
            tsdf_old (numpy.array [v, ]): v is equal to the number of voxels to be
                integrated at this timestamp. Old tsdf values that need to be
                updated based on the current observation.
            margin_distance (numpy.array [v, ]): The tsdf values of the current observation.
                It should be of type numpy.array [v, ], where v is the number
                of valid voxels.
            w_old (numpy.array [v, ]): old weight values.
            observation_weight (float): Weight to give each new observation.

        Returns:
            numpy.array [v, ]: new tsdf values for entries in tsdf_old
            numpy.array [v, ]: new weights to be used in the future.
        """

        """Get the tsdf and color volumes.

        Returns:
            numpy.array [l, w, h]: l, w, h are the dimensions of the voxel grid in voxel space.
                Each entry contains the integrated tsdf value.
            numpy.array [l, w, h, 3]: l, w, h are the dimensions of the voxel grid in voxel space.
                3 is the channel number in the order r, g, then b.
        """

        #get weight_old and color_old through 3d space

        
        weight_old = [] #validx3

        # Create arrays of coordinates
        x_coords = np.array(self._voxel_coords[:,0][bool_valid])
        y_coords = np.array(self._voxel_coords[:,1][bool_valid])
        z_coords = np.array(self._voxel_coords[:,2][bool_valid])

        # for i in range (len(self._voxel_coords)): #nx3


        #     if bool_valid[i] == True: #nx3

        #         weight_old.append(self._weight_volume[self._voxel_coords[i][0]][self._voxel_coords[i][1]][self._voxel_coords[i][2]])
        #         tsdf_old.append(self._tsdf_volume[self._voxel_coords[i][0]][self._voxel_coords[i][1]][self._voxel_coords[i][2]])
        weight_old = self._weight_volume[x_coords, y_coords, z_coords]
        tsdf_old = self._tsdf_volume[x_coords, y_coords, z_coords]
        print("weight_old",np.shape(weight_old))
        print("tsdf_old",np.shape(tsdf_old))
        # for i in range(len(tsdf_old)):
        #     print(tsdf_old[i])

 

        tsdf_new, w_new = self.get_new_tsdf_and_weights(tsdf_old, margin_valid, weight_old, observation_weight)
        print()
        print("tsdf_new",np.shape(tsdf_new))
        print("observe",tsdf_new)
        print("w_new",np.shape(w_new))



        # TODO: 6.
        #color_image (numpy.array [h, w, 3]): An rgb image.
        color_old = self._color_volume[x_coords, y_coords, z_coords]
        print("color_old",np.shape(color_old))
        color_new = color_image[valid_pixel_2d[:,1],valid_pixel_2d[:,0]]
        print("color_new",np.shape(color_new))
        color_neww = TSDFVolume.get_new_colors_with_weights(color_old, color_new, weight_old, w_new, observation_weight)
        
        # TODO: 6.5 Update key parameters self._tsdf_volume, self._weight_volume and self._color_volume
        
        pointer1 = 0
        for i in range (len(self._voxel_coords)):

            if bool_valid[i] == True:
                self._tsdf_volume[self._voxel_coords[i][0]][self._voxel_coords[i][1]][self._voxel_coords[i][2]] = tsdf_new[pointer1]
                self._weight_volume[self._voxel_coords[i][0]][self._voxel_coords[i][1]][self._voxel_coords[i][2]] = w_new[pointer1]
                self._color_volume[self._voxel_coords[i][0]][self._voxel_coords[i][1]][self._voxel_coords[i][2]] = color_neww[pointer1]
                pointer1 += 1
    
    """
    *******************************************************************************
    ******************************* ASSIGNMENT ENDS *******************************
    *******************************************************************************
    """
