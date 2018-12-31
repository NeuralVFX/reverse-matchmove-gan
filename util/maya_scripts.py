import maya.cmds as cmds
import maya.api.OpenMaya as om
import pymel.core as pm
import csv


# Forgive my mixing of cmds and pymel, was
# in a physically awkward AirBnB where it
# was difficult to think straight

def get_mat(cam, w='.worldMatrix[0]'):
    # get matrix from object, world or  object
    wm = cmds.getAttr(cam + w)
    wm_mat = [wm[0:4], wm[4:8], wm[8:12], wm[12:16]]
    trans_mat = om.MMatrix(wm_mat).transpose()
    lm = list(trans_mat)
    tran_list = [lm[0:4], lm[4:8], lm[8:12], lm[12:16]]
    return tran_list


def get_cam(cam):
    # get matrix and focal length from camera
    tran_list = get_mat(cam)
    fv = cmds.getAttr(cam + '.focalLength')
    return tran_list, fv


def export_multi_cam_data(filename='C:/ML/export/dataset_train.csv'):
    # loop through multiple selected cameras and export to csv file
    cam_list = pm.ls(sl=True)

    mat_list = []
    fl_list = []
    image_list = []

    for cam in cam_list:
        # the photomodeler used appended "FBXASC046" to every image name
        # to create the camera name, might be different in other programs
        image = str(cam.name()).replace('FBXASC046', '.')
        mat, fl = get_cam(str(cam))
        image_list.append(image)
        fl_list.append(fl)
        mat_list.append(mat)

    with open(filename, 'wb') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        spamwriter.writerow(['m' + str(mat) for mat in range(16)] + ['focal_length', 'filename'])
        for i in range(len(mat_list)):
            app_mat_list = mat_list[i][0] + mat_list[i][1] + mat_list[i][2] + mat_list[i][3]
            whole_row = app_mat_list + [fl_list[i], image_list[i]]
            row_str = []
            [row_str.append(str(piece)) for piece in whole_row]
            spamwriter.writerow(row_str)
            print ', '.join(row_str)


def new_cam_path_from_existing(frame_offset=100):
    # makes a new camera, which moves through all
    # of the selected cameras as an animation
    cams = cmds.ls(sl=True)

    new_cam = cmds.duplicate(cams[0])

    frame = 1
    for cam in cams:
        x_form = cmds.xform(cam, q=True, m=True, ws=True)
        cmds.currentTime(frame)
        cmds.xform(new_cam, m=x_form, ws=True)
        cmds.setKeyframe(new_cam)
        frame += frame_offset


def export_anim_cam_data(filename='C:/ML/export/dataset_repo.csv', frame_range=200):
    # export single animated camera as a sequence
    # for "REPO" dataset to generate animated GIF
    cam = pm.ls(sl=True)[0]

    mat_list = []
    fl_list = []
    image_list = []

    for frame in range(frame_range):
        cmds.currentTime(frame)
        image = 'C:/None/None.jpg'
        mat, fl = get_cam(str(cam))
        image_list.append(image)
        fl_list.append(fl)
        mat_list.append(mat)

    import csv
    with open(filename, 'wb') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        spamwriter.writerow(['m' + str(mat) for mat in range(16)] + ['focal_length', 'filename'])
        for i in range(len(mat_list)):
            app_mat_list = mat_list[i][0] + mat_list[i][1] + mat_list[i][2] + mat_list[i][3]
            whole_row = app_mat_list + [fl_list[i], image_list[i]]
            row_str = []
            [row_str.append(str(piece)) for piece in whole_row]
            spamwriter.writerow(row_str)
            print ', '.join(row_str)
