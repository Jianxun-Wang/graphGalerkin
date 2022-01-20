import glob
import tempfile
import scipy.misc
import os, sys, subprocess, glob

from pyCaMOtk.check import is_type

def cropbox_from_pixel(A):
    """
    Determine appropriate cropbox from image (pixels format, nx x ny x ncolor)
    by eliminating all pixels on the outer border until the color is different
    than the color in the (0, 0) pixel.

    Input arguments
    ---------------
    A : ndarray, shape (nx, ny, ncolor)
      Image in pixel format

    Return value
    ------------
    xylim : list, size = 4
      Indices such that A[xylim[0]:xylim[1]+1, xylim[2]:xylim[1]+1, :] is the
      cropped image
    """
    if not is_type(A, 'ndarray'):
        raise TypeError('A must be ndarray')
    if not len(A.shape) == 3:
        raise ValueError('A must be 3D array (image in pixel format)')
    ix = (A[:, :, 0] != A[0, 0, 0])
    for ii in range(1, A.shape[2]):
        ix = ix | (A[:, :, ii] != A[0, 0, ii])
    x1min, x1max = ix.any(0).nonzero()[0].min(), \
                   ix.any(0).nonzero()[0].max()
    x0min, x0max = ix.any(1).nonzero()[0].min(), \
                   ix.any(1).nonzero()[0].max()
    return [x0min, x0max, x1min, x1max]

def imgcrop_from_pixel(A):
    """
    Crop image using algorithm in cropbox(*) function.

    Input arguments
    ---------------
    A : ndarray, shape (nx, ny, ncolor)
      Image in pixel format

    Return value
    ------------
    B : ndarray, shape (nx_crop, ny_crop, ncolor)
      Cropped image in pixel format
    """
    box = cropbox(A)
    return A[box[0]:box[1]+1, box[2]:box[3]+1, :]

def mk_cropped_img_from_mpl(fig, fname, dpi=100, resizefac=None,
                            facecolor='w', mksizeeven=False, crop=True):
    """
    Make cropped figure from Matplotlib plot using cropping algorithm
    in cropbox(*).

    Input arguments
    ---------------
    fig : Matplotlib figure
    fname : str
      Filename of output image
    dpi : int
      Dots per inch
    resizefac : None, int, float, tuple
      Resize argument for image: - None  (do not resize),
                                 - int   (percentage of current size)
                                 - float (fraction of current size)
                                 - tuple (size of output image)
    facecolor : str
      Background axis color
    mksizeeven : bool
      Whether to require size of image to be an even number of pixels
    crop : bool
      Whether to crop image (using cropbox() algorithm)

    Example
    -------
    >> # Create figure (fig) using matplotlib
    >> mk_cropped_img_from_mpl(fig, 'tmp.png', 300, None, 'w', False, True)
    """
    if not is_type(fname, 'str'):
        raise TypeError('fname must be str')
    if not is_type(dpi, 'int'):
        raise TypeError('dpi must be int')
    if resizefac is not None and not is_type(resizefac, 'int')    \
                             and not is_type(resizefac, 'number') \
                             and not is_type(resizefac, 'iter_of_int'):
        raise TypeError('resizefac must be None, int, number, or ' + 
                        '\iterable of int')
    if not is_type(facecolor, 'str'):
        raise TypeError('facecolor must be str') 
    if not is_type(mksizeeven, bool):
        raise TypeError('mksizeeven must be bool') 
    if not is_type(crop, bool):
        raise TypeError('crop must be bool') 
    ftemp = tempfile.NamedTemporaryFile().name + '.png'
    fig.savefig(ftemp, dpi=dpi, bbox_inches='tight', pad_inches=.1, \
                facecolor=facecolor)
    A = scipy.misc.imread(ftemp)
    os.remove(ftemp)
    if resizefac is not None:
        A = scipy.misc.imresize(A, resizefac, interp='bicubic')
    if crop:
        A = imgcrop(A)
    if mksizeeven:
        if A.shape[0] % 2 == 1:
            A = A[:-1, :, :]
        if A.shape[1] % 2 == 1:
            A = A[:, :-1, :]
    scipy.misc.imsave(fname, A)

def mk_mp4_from_imglst_ffmpeg(imglst, mp4fn, crf=23, fmt='png',
                              fps_out=24, fps_in=None):
    """
    Make MP4 from list of images using FFMPEG

    Input arguments
    ---------------
    imglst : iterable of str
      Filename of images (ordered) to make into movie
    mp4fn : str
      Filename of output movie
    crf : int (between 0 and 51)
      Constant rate factor (lower value is higher qulify; 23 is default)
    fmt : str
      Image format (only png supported)
    fps_out : int
      Output frames per second
    fps_in : None, int
      Input frames per second

    Examples
    --------
    >> imglst = ['img0.png', 'img1.png']
    >> mk_mp4_from_imglst_ffmpeg(imglst, 'tmp.mp4')
    >> # on implicit
    >> base = '/scratch/simulations/2016_flapopt/flap2d/' + \
              'morph0_rbm_cspline0/img/nacamsh1ref0p3_Re1000_' + \
              'M0p1_Tx0_Fz0_iter070_frame{0:s}.mshmot.png'
    >> imglst = [base.format(str(k).zfill(4)) for k in range(101)]
    >> mk_mp4_from_imglst_ffmpeg(imglst, 'tmp.mp4')
    """
    if not is_type(imglst, 'iter_of_str'):
        raise TypeError('imglst must be iterable of str')
    for img in imglst:
        if len(glob.glob(img)) == 0:
            raise ValueError('Image ' + img + 'does not exist')
    if not is_type(mp4fn, 'str'):
        raise TypeError('mp4fn must be str')
    if not is_type(crf, 'int'):
        raise TypeError('crf must be int')
    if crf < 0 or crf > 51:
        raise ValueError('crf must be between 0, 51')
    if not is_type(fmt, 'str'):
        raise TypeError('fmt must be str')
    if fmt != 'png':
        raise ValueError('fmt must be png for now')
    if not is_type(fps_out, 'int'):
        raise TypeError('fps_out must be int')
    if fps_in is not None and not is_type(fps_in, 'int'):
        raise TypeError('fps_in must be None or int')

    # Create temporary directory
    tmpdir = 'tmpdir_for_im2mp4_deleteme'
    subprocess.call('mkdir {0:s}'.format(tmpdir), shell=True)

    # Copy png into temporary directory
    for imgfn in imglst:
        subprocess.call('cp {0:s} {1:s}/'.format(imgfn, tmpdir),
                        shell=True)

    # Call FFMPEG
    if fps_in is None:
        base = 'ffmpeg -pattern_type glob -i '      + \
               '"{0:s}/*.png" -c:v libx264 '        + \
               '-crf {1:d} -pix_fmt yuv420p -g 24 ' + \
               '-r {2:d} -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" ' + \
               '{0:s}.mp4'
        exec_str = base.format(tmpdir, crf, fps_out)
    else:
        base = 'ffmpeg -r {0:d} -pattern_type glob -i ' + \
               '"{1:s}/*.png" -c:v libx264 -crf {2:d} ' + \
               '-pix_fmt yuv420p -g 24 -r {3:d} '       + \
               '-vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" {1:s}.mp4'
        exec_str = base.format(fps_in, tmpdir, crf, fps_out)
    print exec_str
    subprocess.call(exec_str, shell=True)

    # Move mp4 file
    subprocess.call('mv {0:s}.mp4 {1:s}'.format(tmpdir, mp4fn), shell=True)

    # Delete temporary directory
    subprocess.call('rm {0:s}/*png'.format(tmpdir), shell=True)
    subprocess.call('rmdir {0:s}'.format(tmpdir), shell=True)

if __name__ == '__main__':
    base = '/scratch/simulations/2016_flapopt/flap2d/' + \
           'morph0_rbm_cspline0/img/nacamsh1ref0p3_Re1000_' + \
           'M0p1_Tx0_Fz0_iter070_frame{0:s}.mshmot.png'
    imglst = [base.format(str(k).zfill(4)) for k in range(101)]
    mk_mp4_from_imglst_ffmpeg(imglst, 'tmp.mp4')
