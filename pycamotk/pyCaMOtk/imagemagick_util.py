import subprocess

from pyCaMOtk.check import is_type

def convert_side_by_side(imglst, outimg, prnt_cmd=False, exec_cmd=True):
    """
    Concatenate individual images into a single image (side-by-side)

    Input arguments
    ---------------
    imglst : iterable of str
      Filenames of individual images
    outimg : str
      Filename of output image
    prnt_cmd : bool
      Indicates whether to print command
    exec_cmd : bool
      Indicates whether to execute command

    Return value
    ------------
    exec_str : str
      Imagemagick command

    Example
    -------
    >> convert_side_by_side(['img0.png', 'img1.png'], 'out.png', False, False)
    # 'convert img0.png img1.png +append out.png'
    """
    if not is_type(imglst, 'iter_of_str'):
        raise TypeError('imglst must be iterable of str')
    if not is_type(outimg, 'str'):
        raise TypeError('outimg must be str')
    if not is_type(prnt_cmd, bool):
        raise TypeError('prnt_cmd must be bool')
    if not is_type(exec_cmd, bool):
        raise TypeError('exec_cmd must be bool')
    exec_str = 'convert '
    for img in imglst:
        exec_str += img + ' '
    exec_str += '+append ' + outimg
    if prnt_cmd: print(exec_str)
    if exec_cmd : subprocess.call(exec_str, shell=True)
    return exec_str

def montage_grid(imglstlst, outimg, prnt_cmd=False, exec_cmd=True, img2d={}):
    """
    Concatenate individual images into a grid image (matrix). The shape of
    imglstlst determines configuration of grid in output image. The image
    imglstlst[i][j] will be in row i, col j of the output image.

    Input arguments
    ---------------
    imglstlst : iterable of iterable of str, size = (m, n)
      Filenames of individual images
    outimg : str
      Filename of output image
    prnt_cmd : bool
      Indicates whether to print command
    exec_cmd : bool
      Indicates whether to execute command
    img2d : dict
      Options to pass to montage
       - border : iterable of int, size 2
           Horizontal, vertical spacing between images, respectively

    Return value
    ------------
    exec_str : str
      Imagemagick command

    Example
    -------
    >> imglstlst = [['img00.png', 'img01.png'], ['img10.png', 'img11.png']]
    >> montage_grid(imglstlst, 'out.png', False, False)
    # 'montage img00.png img01.png img10.png img11.png -tile 2x2 ' + \
      '-geometry +10+10 out.png'
    """
    if not is_type(imglstlst, 'iter_of_iter_of_str'):
        raise TypeError('imglstlst must be iterable of iterable of str')
    if not is_type(outimg, 'str'):
        raise TypeError('outimg must be str')
    if not is_type(prnt_cmd, bool):
        raise TypeError('prnt_cmd must be bool')
    if not is_type(exec_cmd, bool):
        raise TypeError('exec_cmd must be bool')
    m, n = len(imglstlst), len(imglstlst[0])
    for k in range(1, m):
        if len(imglstlst[k]) != n:
            raise ValueError('All rows of imglstlst must have same ' + \
                             'number of columns')
    border = img2d['border'] if 'border' in img2d else (10, 10)
    exec_str = 'montage '
    for imglst in imglstlst: exec_str += ' '.join(imglst) + ' '
    exec_str += '-tile {0:d}x{1:d} '.format(n, m)
    exec_str += '-geometry +{0:d}+{1:d} '.format(border[0], border[1])
    exec_str += outimg
    if prnt_cmd: print(exec_str)
    if exec_cmd : subprocess.call(exec_str, shell=True)
    return exec_str
