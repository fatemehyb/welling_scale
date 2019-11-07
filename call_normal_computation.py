# mesh created with 
# verts, faces = skimage.measure.marching_cubes(volume, level, spacing=(1.0, 1.0, 1.0))

from mayavi import mlab
# import os
# import argparse
# import itk
# import SimpleITK as sitk
# import numpy as np
# import read_image_m as RIM
import glm

from skimage.measure import marching_cubes_lewiner
X_2=[]
Y_2=[]
Z_2=[]

def Cross(a, b):
    x = a.y * b.z - a.z * b.y
    y = a.z * b.x - a.x * b.z
    z = a.x * b.y - a.y * b.x

    # //    return (a.y*b.z-a.z*b.y,a.z*b.x-a.x*b.z,a.x*b.y-a.y*b.x);
    return glm.vec3(x, y, z)
def Dot(a, b):
    x = a.x * b.x
    y = a.y * b.y
    z = a.z * b.z

    # //    return (a.y*b.z-a.z*b.y,a.z*b.x-a.x*b.z,a.x*b.y-a.y*b.x);
    return glm.vec3(x, y, z)



def main_normal(myvolume,spacing):
        # verts, faces = skimage.measure.marching_cubes(volume, level, spacing=(1,1,1))
        verts, faces, normals, values = marching_cubes_lewiner(myvolume, 400.0, spacing)
        mesh=mlab.triangular_mesh([vert[0] for vert in verts],
                             [vert[1] for vert in verts],
                             [vert[2] for vert in verts],
                             faces)




        # A first plot in 3D
        fig = mlab.figure(1)


        cursor3d = mlab.points3d(0., 0., 0., mode='axes',
                                        color=(0, 0, 0),
                                        scale_factor=0.5)
        mlab.title('Click on the volume to determine 3 points(consider right hand rule)')



        ################################################################################
        # Some logic to select 'mesh' and the data index when picking.

        def picker_callback(picker_obj):
            picked = picker_obj.actors
            if mesh.actor.actor._vtk_obj in [o._vtk_obj for o in picked]:
                # m.mlab_source.points is the points array underlying the vtk
                # dataset. GetPointId return the index in this array.
                # x_, y_ = np.lib.index_tricks.unravel_index(picker_obj.point_id, s.shape)
                x_2, y_2, z_2 = picker_obj.pick_position
                # x_2, y_2, z_2 = picker_obj.mapper_position
                X_2.append(x_2/spacing[0])
                Y_2.append(y_2/spacing[1])
                Z_2.append(z_2/spacing[2])

                print("Data indices: %f, %f, %f" % (x_2/spacing[0], y_2/spacing[1], z_2/spacing[2]))




        fig.on_mouse_pick(picker_callback)

        mlab.show()
        p1 = glm.vec3(X_2[0],Y_2[0],Z_2[0])
        p2 = glm.vec3(X_2[1],Y_2[1],Z_2[1])
        p3 = glm.vec3(X_2[2],Y_2[2],Z_2[2])

        # These two vectors are in the plane
        v1 = glm.vec3(p3.x - p1.x,p3.y-p1.y,p3.z-p1.z)
        v2 = glm.vec3(p2.x - p1.x,p2.y-p1.y,p2.z-p1.z)

        # the cross product is a vector normal to the plane
        The_Normal =glm.normalize(Cross(v1, v2))
        The_Normal2=The_Normal
        The_Normal2.x=The_Normal.y
        The_Normal2.y=(The_Normal.x)
        The_Normal2.z=(The_Normal.z)
        # The_Normal3=glm.triangleNormal(p1,p2,p3)
        print("Normal is: ")
        print((The_Normal2))
        # print(The_Normal3)


        return ((The_Normal2),p1,p2,p3)

        # From:
        # http://scikit-image.org/docs/dev/api/skimage.measure.html?highlight=marching_cubes#skimage.measure.marching_cubes
        # http://scikit-image.org/docs/dev/auto_examples/plot_marching_cubes.html
