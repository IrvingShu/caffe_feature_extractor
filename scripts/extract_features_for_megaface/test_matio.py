from matio import read_mat, save_mat


def test(mat_bin_fn):
    print '===> load feat from bin file:', mat_bin_fn
    f = open(mat_bin_fn, 'rb')
    feat = read_mat(f)
    f.close()
    print 'feat.shape:', feat.shape
    print 'feat: ', feat

    save_fn = mat_bin_fn + '_resaved.bin'
    print '===> resave feat into bin file:', save_fn
    save_mat(save_fn, feat)

    print '===> load feat from bin file:', save_fn
    f = open(save_fn, 'rb')
    feat2 = read_mat(f)
    f.close()
    print 'resaved feat2.shape:', feat2.shape

    diff = feat -feat2
    print '===> sum(feat-feat2) = ', diff.sum()


if __name__ == '__main__':
    mat_bin_fn = './test_feat/Chris_Evans_10912.png_LBP_10x10.bin'
    test(mat_bin_fn)
